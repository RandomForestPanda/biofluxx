# NDVI-Enhanced Predator-Prey Ecosystem Simulation with Reinforcement Learning

![Project Banner](public/static/keyur-nandaniya-5xxA9OS4r3s-unsplash.jpeg)  
*A visualization of dynamic ecosystem interactions in a predator-prey model integrated with real-world NDVI data.*

## Project Overview

This project is a sophisticated simulation of a predator-prey ecosystem enhanced with Normalized Difference Vegetation Index (NDVI) data from satellite imagery (TIFF files). It models the interactions between herbivores (prey) and carnivores (predators) in a grid-based environment where vegetation growth influences herbivore behavior. The simulation incorporates optional Reinforcement Learning (RL) for herbivores, allowing them to learn optimal movement strategies based on NDVI values and energy management.

The core simulation is built in Python using scientific libraries like NumPy, SciPy, and Matplotlib. A user-friendly web interface, powered by FastAPI, enables interactive configuration, real-time visualization, and result analysis. This project demonstrates advanced concepts in ecological modeling, agent-based simulation, reinforcement learning, and web development.

As a major project, it showcases my expertise in:
- Integrating geospatial data (NDVI from TIFF files) into dynamic simulations.
- Implementing agent-based models with diffusion, reproduction, and predation mechanics.
- Applying Q-learning for adaptive agent behavior.
- Building responsive web applications for scientific visualization.

This simulation can be used for educational purposes (e.g., teaching Lotka-Volterra dynamics), environmental research (e.g., studying vegetation impact on wildlife), or as a foundation for more complex AI-driven ecological models.

## Key Features

- **NDVI Integration**: Loads and normalizes NDVI data from multiple TIFF files, updating the vegetation grid dynamically with logistic growth and Gaussian diffusion.
- **Agent-Based Modeling**:
  - Herbivores consume vegetation, reproduce, and move towards high-NDVI areas.
  - Carnivores hunt herbivores, reproduce, and move towards high-prey density.
  - Energy management system for survival and reproduction.
- **Reinforcement Learning Option**: Herbivores can use Q-learning to optimize actions (move north/south/east/west or stay) based on states like position, NDVI, and energy.
- **Diffusion Mechanics**: Gaussian filtering simulates population spread and vegetation regrowth.
- **Visualization**:
  - Real-time animation of NDVI, herbivore, and carnivore grids using Matplotlib.
  - Population and NDVI history plots.
  - Interactive web dashboard with Chart.js for live charts and Plotly for advanced graphs.
  - Combined static plots for NDVI and population distributions.
- **Web Interface**:
  - FastAPI backend for simulation control and API endpoints.
  - Responsive HTML templates with Jinja2 for landing and simulation pages.
  - 3D visualization placeholders for future ecosystem rendering (using Three.js).
- **Customization**: Users can adjust parameters like animal counts, simulation steps, NDVI update frequency, and RL toggling via the web UI.
- **Output Options**: Save animations as MP4, export results as JSON, and generate static plots.

## Technologies Used

- **Core Simulation**:
  - Python 3.12+
  - NumPy & SciPy: For grid operations, diffusion, and NDVI processing.
  - Matplotlib & Animation: For real-time visualizations and animations.
  - TiffFile: For loading geospatial TIFF data.
  - Plotly: For interactive web-based graphs.
- **Reinforcement Learning**:
  - Custom Q-learning implementation with epsilon-greedy exploration.
- **Web Framework**:
  - FastAPI: High-performance API for simulation control.
  - Jinja2: Templating for dynamic HTML.
  - Uvicorn: ASGI server for running the app.
- **Frontend**:
  - HTML5/CSS3: Responsive design with CSS variables for theming.
  - Chart.js & Plotly.js: For charts and plots.
  - SweetAlert2: For user notifications.
  - Three.js: For potential 3D visualizations (integrated in templates).
- **Other Tools**:
  - Requests: For downloading 3D assets.
  - OS/Pathlib: For file handling.
  - JSON/NumpyEncoder: For serializing simulation data.

## Architecture Overview

The project is divided into two main components:

1. **Simulation Core (`main_rl_eg.py`)**:
   - `NDVIPredatorPreySimulation` class: Handles NDVI loading, agent initialization, updates, and RL logic.
   - Methods: `update_ndvi()`, `move_herbivores()` (with RL), `move_carnivores()`, `run_simulation()`.
   - Outputs: Grid states, population histories, and Plotly graphs.

2. **Web Interface (`webinterface.py`)**:
   - FastAPI app with endpoints: `/start_simulation`, `/get_results`, `/get_plot_data`, `/get_combined_plot`.
   - Templates: `landing.html` (home page with features), `index.html` (simulation dashboard).
   - Static Files: CSS styles, images, and 3D models (e.g., deer, wolf GLTF files).

**Data Flow**:
- User inputs parameters via web form → FastAPI starts simulation → Results stored in global object → Frontend fetches and visualizes data.

## Installation

### Prerequisites
- Python 3.12+
- Git
- Virtual environment tool (e.g., venv)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ndvi-predator-prey-simulation.git
   cd ndvi-predator-prey-simulation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `requirements.txt` is not provided, install manually:*
   ```bash
   pip install numpy scipy matplotlib tifffile fastapi uvicorn jinja2 requests plotly chart.js sweetalert2
   ```

4. Prepare NDVI Data:
   - Place TIFF files in the `data_tiff/` folder (dummy files are generated if missing).
   - Ensure TIFF files contain valid NDVI data (values > -999).

5. Run the Application:
   ```bash
   python webinterface.py
   ```
   - Access the web interface at `http://localhost:8000/`.

## Usage

### Running the Simulation via Web Interface
1. Navigate to `http://localhost:8000/` for the landing page.
2. Click "Launch Simulation" to go to the dashboard.
3. Configure parameters (e.g., herbivores: 1000, carnivores: 200, steps: 100, enable RL).
4. Click "Start Simulation".
5. View live charts, refresh data, or show combined plots.

### Running Standalone Simulation
From `main_rl_eg.py`:
```python
sim = NDVIPredatorPreySimulation(tiff_folder_path="data_tiff", num_herbivores=1500, num_carnivores=500, rl_herbivores=True)
sim.run_simulation(steps=300, animate=True, save_animation=True)
sim.plot_results()
```

### API Endpoints
- `POST /start_simulation`: Start simulation with JSON body (e.g., `{ "num_herbivores": 1000 }`).
- `GET /get_results`: Fetch simulation history as JSON.
- `GET /get_combined_plot`: Get PNG of combined visualizations.

## Demo & Screenshots

- **Landing Page**:  
  ![Landing Page](public/static/screenshot-landing.png)  
  *Features overview with calls-to-action.*

- **Simulation Dashboard**:  
  ![Simulation Dashboard](public/static/screenshot-dashboard.png)  
  *Interactive controls and real-time charts.*

- **Animation Output**:  
  Run with `animate=True` to see dynamic grids (saved as `ndvi_predator_prey_time.mp4`).

- **Plotly Results**:  
  Interactive graphs showing population dynamics and NDVI means.

For a live demo, deploy to a server like Heroku or Vercel.

## Future Work
- Integrate real-time 3D rendering with Three.js for immersive ecosystem views.
- Add more RL agents (e.g., for carnivores) using advanced»

libraries like Stable Baselines3.
- Support uploading custom TIFF files via the web UI.
- Implement multi-agent RL with shared Q-tables.
- Add environmental factors like climate change impacts on NDVI.
- Deploy as a Docker container for easier distribution.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by classic Lotka-Volterra models and geospatial ecology.
- 3D models from Khronos Group's glTF Sample Models (CC-BY-4.0).
- Thanks to open-source libraries like FastAPI, Matplotlib, and SciPy for enabling this project.

*Developed by [Your Name] as a major portfolio project. Contact: your.email@example.com | GitHub: @yourusername*