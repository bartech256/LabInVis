# GeoGNN: Modeling Spatial Relational Real Estate Pricing with Graph Neural Networks  

## Project Overview
GeoGNN is a Python-based project designed to model real estate pricing using Graph Neural Networks (GNNs).  
It leverages spatial and structural relationships between properties to improve predictive accuracy. 

This project explores real estate price prediction using Graph Neural Networks (GNNs).
We model housing as a graph where:

Nodes represent properties (with structural, spatial, and engineered features).

Edges represent spatial relationships between properties, grouped into humanistic distance categories (walking, biking, driving).

We compare different GNN architectures against strong baselines such as CatBoost to investigate whether graph-based models improve prediction accuracy.

Key features include:
- Graph construction based on geographic distance and property similarity.
- Multiple GNN architectures: SimpleGCN, SimpleGAT, MultiLayerGCN, GraphSAGE.
- Baseline regression models: MLPRegressor, LinearRegressionTorch.
- Combined GNN + regression models.
- Modular experiment pipeline with configurable settings.
- Training and evaluation loops with scalable metrics.

## Setup Instructions

There are several ways to run this project:

1. Download the files:
If you reached this github link, all you have to do is download the files as zip.
Then you extract it, enter the main code via your preferred app (like pycharm),
and run the README code below.

2. Repository URL:
You can clone this repository using:  
```bash
git clone https://github.com/bartech256/LabInVis.git
cd LabInVis
```

2.1
Install dependencies:
```bash
pip install -r requirements.txt
```


## Planned Usage
Experiments will be run via:
```bash
python main.py --config-dir .\configs