import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams
# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
SEQ_CACHE = {}
PROJECTIONS = {}

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas, z_norms = execute_alphaedit(model, tok, requests, hparams, cache_template=cache_template)
    distances = {}
    distances['z_norms'] = z_norms

    with torch.no_grad():
        for w_name, (upd_matrix, inside_norms) in deltas.items():

            upd_matrix = upd_matrix.to("cuda")

            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            original_weights_norm = torch.norm(w[...]).detach().cpu().item()

            w[...] += upd_matrix.float()


            #cap singular value to 11
            sv_cap = 11
            U, S, Vh = torch.linalg.svd(w[...], full_matrices=False) 
            S_clipped = torch.clamp(S, max=sv_cap)
            #w[...] =  U @ torch.diag(S_clipped) @ Vh

            #rescale norm
            new_weights_norm = torch.norm(w[...]).detach().cpu().item()
            alpha = original_weights_norm/new_weights_norm
            #w[...] *= alpha

            #spectral normalization
            gamma = 10.2656
            _, S, _ = torch.linalg.svd(w[...], full_matrices=False)
            spectral_norm = torch.max(S)
            #w[...] = gamma * (w[...] / spectral_norm)


            start = time.time()
            _, svd_upd, _ = torch.linalg.svd(upd_matrix)
            svd_upd = sorted(svd_upd.detach().cpu().tolist(), reverse=True)
            
            _, svd_final, _ = torch.linalg.svd(w[...])
            svd_final = sorted(svd_final.detach().cpu().tolist(), reverse=True)
            print('svd calculation time:', time.time() - start)


            #saving all distances
            layer = w_name.split('.')[2]
            temp_dict = {
                'preservation_distance': None,
                'new_edit_distance': None,
                'old_edit_distance': None,
                'delta_norm': torch.norm(upd_matrix).detach().cpu().item(),
                'new_weights_norm': torch.norm(w[...]).detach().cpu().item(),
                'original_weights_norm': original_weights_norm,
                'inside_norms': inside_norms,
                'alpha': alpha,
                'svd_final': svd_final,
                'svd_upd': svd_upd, 
            }
            distances[layer] = temp_dict

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    #return all the objective loss terms, plus the absolute norm of the new weights

    return model, weights_copy, distances




def execute_alphaedit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    z_norms = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration. 
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for r_id, request in enumerate(requests):
        print(r_id)
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z, delta_norm, init_norm = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )
            z_norms[r_id] = {'delta': delta_norm, 'init_norm': init_norm, 'final_norm': cur_z.norm().item()}
            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)


    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        
        #pre-compute projection layers
        if layer not in PROJECTIONS:
            PROJECTIONS[layer] = get_project(model,tok,layer,hparams)

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)



        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers

        ##Store previous sequential
        if layer not in SEQ_CACHE:
            SEQ_CACHE[layer] = (layer_ks @ layer_ks.T).cpu()
        else:
            SEQ_CACHE[layer] += (layer_ks @ layer_ks.T).cpu()

        upd_matrix = torch.linalg.solve(
                PROJECTIONS[layer].cuda() @ SEQ_CACHE[layer].cuda() + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), PROJECTIONS[layer].cuda() @ layer_ks @ resid.T
        )

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        ##calculate_norms
        inside_norms = {
            'zs_norm' : torch.mean(torch.norm(zs, dim = 0)).detach().cpu().item()
        }
        
        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()

            deltas[weight_name] = (
                upd_matrix.cpu(),
                inside_norms
            )

        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x

        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas, z_norms




def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    model_name = model_name.split('_')[-1]
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat, preserved_keys = layer_stats(
                model,
                tok,
                layer_name,
                STATS_DIR,
                mom2_dataset,
                to_collect=["mom2"],
                sample_size=mom2_n_samples,
                precision=mom2_dtype,
                force_recompute=force_recompute,
            )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
