# 🗺️ TravelTimeApp

**TravelTimeApp** is an interactive Python-based tool that visualizes *multi-mode travel-time accessibility maps* for user-defined Points of Interest (POIs).  
It supports travel modes such as **walk**, **bike**, and **drive**, and automatically generates reachability maps based on travel time, departure time, and mode selection.

The app integrates a powerful **geospatial analysis pipeline** (using `OSMnx`, `GeoPandas`, and `Contextily`) with an intuitive **Tkinter-based user interface**, allowing anyone to run complex spatial computations without writing code.

---

## 🚀 Features

- 🧭 **Multi-mode travel analysis**: Supports `walk`, `bike`, `drive`, and their combinations.
- 🏙️ **User-defined POIs**: Input three POIs (locations or landmarks) separated by commas.
- ⏱️ **Travel time control**: Choose a departure time (in minutes) to define reachability.
- 🗺️ **Automatic map generation**: Generates and saves high-resolution accessibility maps.
- 💡 **Progress bar feedback**: Displays live progress during data processing.
- 🖼️ **Interactive gallery viewer**: View all generated maps inside the app (Prev / Next / Auto-play).
- 📂 **One-click output access**: “Open Maps Folder” button opens the results directly.
- 💻 **Standalone executable**: Distributed as a single `.exe` for Windows users — no installation required.

---

## 🧩 System Requirements

- **Operating System:** Windows 10/11 (tested), macOS and Linux supported via source.
- **Python version:** 3.9 – 3.11  
- **Required packages:**
  ```bash
  osmnx
  geopandas
  contextily
  shapely
  fiona
  pyproj
  rtree
  matplotlib
  pillow
  tkinter (built-in)
  scikit-learn

**If you plan to run the app from source, it’s best to use a Conda environment.**

## Installation (From Source)
1. Clone the repository:
   ```git clone git@github.com:yanjie99/TravelTimeAnalysisTool.git
      cd TravelTimeApp```


