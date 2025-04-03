
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import os
from main_rl_eg import NDVIPredatorPreySimulation
from typing import Optional
import uvicorn
import numpy as np
import json
import matplotlib.pyplot as plt
import io
import base64
import requests
from pathlib import Path
import matplotlib

app = FastAPI()
matplotlib.use('Agg') 

# Create static directory if it doesn't exist
Path("static").mkdir(exist_ok=True)

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

templates = Jinja2Templates(directory="templates")


app.mount("/static", StaticFiles(directory="public/static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/models", StaticFiles(directory="public/models"), name="models")

# Global simulation object
simulation: Optional[NDVIPredatorPreySimulation] = None



def download_3d_model(url, filename):
    """Download a 3D model from a URL if it doesn't exist locally"""
    filepath = f"static/{filename}"
    if not os.path.exists(filepath):
        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
    return f"/static/{filename}"

def get_landing_page_3d_assets():
    """Get URLs for 3D models used in landing page"""
    return {
        "deer_model": download_3d_model(
            "https://github.com/KhronosGroup/glTF-Sample-Models/raw/master/2.0/Deer/glTF/Deer.gltf",
            "deer.gltf"
        ),
        "wolf_model": download_3d_model(
            "https://github.com/KhronosGroup/glTF-Sample-Models/raw/master/2.0/Wolf/glTF/Wolf.gltf",
            "wolf.gltf"
        ),
        "tree_model": download_3d_model(
            "https://github.com/KhronosGroup/glTF-Sample-Models/raw/master/2.0/Tree/glTF/Tree.gltf",
            "tree.gltf"
        ),
        "terrain_texture": download_3d_model(
            "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/MetalRoughSpheres/glTF/MetalRoughSpheres_baseColor.png",
            "grass_texture.png"
        )
    }

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Render the landing page with 3D visualization"""
    assets = get_landing_page_3d_assets()
    return templates.TemplateResponse("landing.html", {
        "request": request,
        "assets": assets
    })

@app.get("/simulation", response_class=HTMLResponse)
async def simulation_page(request: Request):
    """Render the main simulation page"""
    assets = get_landing_page_3d_assets()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "assets": assets
    })

@app.post("/start_simulation")
async def start_simulation(request: Request):
    """Start the simulation with user-provided parameters."""
    global simulation
    data = await request.json()
    
    tiff_folder = data.get('tiff_folder', 'data_tiff')
    num_herbivores = int(data.get('num_herbivores', 1000))
    num_carnivores = int(data.get('num_carnivores', 200))
    steps = int(data.get('steps', 100))
    rl_herbivores = bool(data.get('rl_herbivores', False))

    # Initialize simulation
    simulation = NDVIPredatorPreySimulation(
        tiff_folder_path=tiff_folder,
        num_herbivores=num_herbivores,
        num_carnivores=num_carnivores,
        ndvi_update_frequency=10,
        rl_herbivores=rl_herbivores
    )

    # Run simulation
    simulation.run_simulation(steps=steps, animate=False, save_animation=False)

    return JSONResponse({"message": "Simulation completed successfully!"})

@app.get("/get_results")
async def get_results():
    """Return the results of the simulation."""
    global simulation
    if not simulation:
        raise HTTPException(status_code=400, detail="Simulation not started yet.")

    results = {
        'herbivore_count_history': convert_to_serializable(simulation.herbivore_count_history),
        'carnivore_count_history': convert_to_serializable(simulation.carnivore_count_history),
        'ndvi_mean_history': convert_to_serializable(simulation.ndvi_mean_history)
    }
    
    return JSONResponse(content=results)

def convert_to_serializable(data):
    """Convert numpy arrays and types to Python native types."""
    if isinstance(data, np.ndarray):
        return data.astype(float).tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    return data

@app.get("/get_plot_data")
async def get_plot_data():
    """Provide data for interactive Plotly plots."""
    global simulation
    if not simulation:
        raise HTTPException(status_code=400, detail="Simulation not started yet.")

    plot_data = {
        "herbivore_count_history": simulation.herbivore_count_history,
        "carnivore_count_history": simulation.carnivore_count_history,
        "ndvi_mean_history": simulation.ndvi_mean_history
    }

    # Convert all numpy arrays to lists and numpy types to native Python types
    converted_data = {}
    for key, value in plot_data.items():
        converted_data[key] = convert_to_serializable(value)
    
    return JSONResponse(content=converted_data, media_type="application/json")

@app.get("/get_combined_plot")
async def get_combined_plot():
    """Generate and serve a combined plot of all visualizations"""
    global simulation
    
    if simulation is None:
        raise HTTPException(
            status_code=400,
            detail="Simulation not initialized. Please start the simulation first."
        )
    
    # Create directory if it doesn't exist
    os.makedirs("static/plots", exist_ok=True)
    plot_file = 'static/plots/combined_plot.png'
    
    try:
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot NDVI
        ndvi_plot = ax1.imshow(simulation.ndvi_grid, cmap='YlGn', vmin=0, vmax=1)
        ax1.set_title('NDVI Distribution')
        fig.colorbar(ndvi_plot, ax=ax1, label='NDVI Value')
        
        # Plot Herbivores
        herb_vmax = max(1, np.max(simulation.herbivore_grid))
        herb_plot = ax2.imshow(simulation.herbivore_grid, cmap='Blues', vmin=0, vmax=herb_vmax)
        ax2.set_title('Herbivore Distribution')
        fig.colorbar(herb_plot, ax=ax2, label='Population Density')
        
        # Plot Carnivores
        carn_vmax = max(1, np.max(simulation.carnivore_grid))
        carn_plot = ax3.imshow(simulation.carnivore_grid, cmap='Reds', vmin=0, vmax=carn_vmax)
        ax3.set_title('Carnivore Distribution')
        fig.colorbar(carn_plot, ax=ax3, label='Population Density')
        
        # Add overall title
        fig.suptitle('Ecosystem Simulation Results', fontsize=16, y=1.05)
        
        # Save and close
        plt.tight_layout()
        plt.savefig(plot_file, bbox_inches='tight', dpi=100)
        plt.close()
        
        return FileResponse(plot_file, media_type='image/png')
        
    except Exception as e:
        if 'plt' in locals():
            plt.close()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate combined plot: {str(e)}"
        )
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


