
from travel_time_app.pipeline import run_pipeline

if __name__ == "__main__":
    LOCATION = "Travel Time Analysis (SG)"
    POI_NAMES = ["NUS University Town", "Harbourlights", "HortPark Singapore"]
    MODES = ["walk", "bike", "drive"]
    DURATIONS = [15]

    run_pipeline(location_name=LOCATION,
                 poi_inputs=POI_NAMES,
                 modes=MODES,
                 durations_min=DURATIONS,
                 save_figs=True,
                 export_vectors=False)
