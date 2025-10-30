# TravelTimeAnalysis2.py
# Wrapper layer adding progress reporting to the user's original TravelTimeAnalysis.py
# Place this file in the SAME directory as TravelTimeAnalysis.py

import os
import time
from typing import Callable, Optional, Iterable, Dict, Any

# Force headless rendering to avoid extra windows
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

try:
    import TTA_SG as TTA  # the user's original pipeline
except Exception as e:
    raise RuntimeError(f"[TTA_SG2] Failed to import TTA_SG.py next to this file: {e}")

# Expose SAFE if present (used by UI code to build filenames)
SAFE = getattr(TTA, 'SAFE', lambda s: ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(s)))

class _Progress:
    def __init__(self, cb: Optional[Callable[[int, str], None]] = None):
        self.cb = cb or (lambda p, m: None)
    def set(self, p: int, msg: str = ''):
        p = max(0, min(100, int(p)))
        try:
            self.cb(p, msg)
        except Exception:
            pass

def _supports_progress_kw(func) -> bool:
    try:
        import inspect
        sig = inspect.signature(func)
        return 'progress' in sig.parameters
    except Exception:
        return False

def _try_wrap_internal_progress(P: _Progress) -> Dict[str, Any]:
    # Best-effort: if the original module has certain known functions,
    # wrap them to emit progress at key milestones.
    wrapped = {}

    # resolve_pois
    if hasattr(TTA, 'resolve_pois'):
        _orig_resolve = TTA.resolve_pois
        def _resolve_pois_wrapped(*args, **kwargs):
            P.set(5, 'Geocoding POIs')
            out = _orig_resolve(*args, **kwargs)
            try:
                n = len(out) if out is not None else 0
            except Exception:
                n = 0
            P.set(15, f'POIs resolved: {n}')
            return out
        TTA.resolve_pois = _resolve_pois_wrapped
        wrapped['resolve_pois'] = True

    # download_graph
    if hasattr(TTA, 'download_graph'):
        _orig_download = TTA.download_graph
        _dl_count = {'n': 0}
        def _download_graph_wrapped(*args, **kwargs):
            _dl_count['n'] += 1
            step = min(20, 5 + _dl_count['n'] * 2)
            P.set(15 + step, f'Downloading OSM graph ({_dl_count["n"]})')
            out = _orig_download(*args, **kwargs)
            return out
        TTA.download_graph = _download_graph_wrapped
        wrapped['download_graph'] = True

    # nearest_node_no_sklearn
    if hasattr(TTA, 'nearest_node_no_sklearn'):
        _orig_nn = TTA.nearest_node_no_sklearn
        _nn_count = {'n': 0}
        def _nn_wrapped(*args, **kwargs):
            _nn_count['n'] += 1
            step = min(5, _nn_count['n'])
            P.set(35 + step, 'Locating nearest nodes')
            return _orig_nn(*args, **kwargs)
        TTA.nearest_node_no_sklearn = _nn_wrapped
        wrapped['nearest_node_no_sklearn'] = True

    # get_reachable_subgraph
    if hasattr(TTA, 'get_reachable_subgraph'):
        _orig_reach = TTA.get_reachable_subgraph
        _reach_count = {'n': 0}
        def _reach_wrapped(*args, **kwargs):
            _reach_count['n'] += 1
            step = min(45, 10 + _reach_count['n'] * 3)
            P.set(40 + step, 'Computing reachability')
            return _orig_reach(*args, **kwargs)
        TTA.get_reachable_subgraph = _reach_wrapped
        wrapped['get_reachable_subgraph'] = True

    # plot_single_mode_frontiers
    if hasattr(TTA, 'plot_single_mode_frontiers'):
        _orig_plot_single = TTA.plot_single_mode_frontiers
        _plot_single_count = {'n': 0}
        def _plot_single_wrapped(*args, **kwargs):
            _plot_single_count['n'] += 1
            step = min(7, _plot_single_count['n'])
            P.set(85 + step, 'Rendering per-mode figure')
            return _orig_plot_single(*args, **kwargs)
        TTA.plot_single_mode_frontiers = _plot_single_wrapped
        wrapped['plot_single_mode_frontiers'] = True

    # plot_multi_mode_overlay
    if hasattr(TTA, 'plot_multi_mode_overlay'):
        _orig_plot_multi = TTA.plot_multi_mode_overlay
        def _plot_multi_wrapped(*args, **kwargs):
            P.set(92, 'Rendering multimode overlay')
            return _orig_plot_multi(*args, **kwargs)
        TTA.plot_multi_mode_overlay = _plot_multi_wrapped
        wrapped['plot_multi_mode_overlay'] = True

    return wrapped

def run_pipeline(
    location_name: str,
    poi_inputs: Iterable,
    modes: Iterable[str] = ('walk', 'bike', 'drive'),
    durations_min: Iterable[float] = (15,),
    speeds_kmh: Optional[Dict[str, float]] = None,
    node_penalty_min: Optional[float] = None,
    save_figs: bool = True,
    export_vectors: bool = False,
    progress: Optional[Callable[[int, str], None]] = None,
) -> None:
    # Wrapper entrypoint. Emits progress even if the original pipeline
    # does not support it (coarse), and tries to inject finer updates when possible.
    P = _Progress(progress)
    P.set(1, 'Init')

    # If the original function supports 'progress', prefer direct call
    if _supports_progress_kw(TTA.run_pipeline):
        P.set(2, 'Delegating with progress')
        return TTA.run_pipeline(
            location_name=location_name,
            poi_inputs=poi_inputs,
            modes=modes,
            durations_min=durations_min,
            speeds_kmh=speeds_kmh,
            node_penalty_min=node_penalty_min,
            save_figs=save_figs,
            export_vectors=export_vectors,
            progress=progress,
        )

    # Otherwise, attempt to wrap a few known functions to emit updates
    wrapped = _try_wrap_internal_progress(P)

    try:
        P.set(3, 'Starting pipeline')
        TTA.run_pipeline(
            location_name=location_name,
            poi_inputs=poi_inputs,
            modes=modes,
            durations_min=durations_min,
            speeds_kmh=speeds_kmh,
            node_penalty_min=node_penalty_min,
            save_figs=save_figs,
            export_vectors=export_vectors,
        )
        if not wrapped:
            P.set(60, 'Processing...')
            time.sleep(0.05)
            P.set(90, 'Rendering...')
        P.set(100, 'Done')
    except Exception as e:
        P.set(0, 'Error')
        raise
    finally:
        pass