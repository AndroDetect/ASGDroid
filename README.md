# ASGDroid

ASGDroid is a project for Android malware detection and malicious component localization that generates Abstract Sensitive Graphs (ASG) for malware analysis by extracting permissions, call graphs, and taint paths.

## Project Structure

```
ASGDroid/
├── src/                           # Source code directory
│   ├── extract_permission/        # Permission extraction module
│   │   └── extract_permission.py  # Permission extraction script
│   ├── extract_call_graph/        # Call graph extraction module
│   │   ├── extract_call_graphs.py # Call graph extraction script
│   │   ├── Appgraph.java          # Java call graph generation class
│   │   ├── Appgraph.class         # Compiled Java class file
│   │   └── parseGraph.py          
│   ├── extract_taint_path/        # Taint path extraction module
│   │   ├── taint_analysis.py      # Taint analysis script
│   │   └── parseXML.py            # Parsing XML file to simplified paths
│   ├── absGraph/                  # ASG generation module
│   │   └── generate_absGraph.py   # ASG generation script
│   ├── nodeRepresentation/        # Node representation module
│   ├── train_and_test/            # Training and testing module
│   └── input/                     # Input files for modules
├── Dataset/                       # Dataset directory
│   └── 2022/                      # 2022 dataset (a sample)
├── Raw/                          # Raw data directory
│   └── 2022/                      # 2022 raw data
│       ├── call_graph/            # Call graph data
│       ├── taint_path/            # Taint path data
│       └── permission/            # Permission data
├── Features/                      # Feature data directory
│   └── 2022/                      # 2022 feature data (ASG with permission)
├── Output/                        # Output results
├── Lib/                          # Dependencies library directory
│   ├── platforms/                 # Android platforms
│   └── soot-infoflow-cmd-jar-with-dependencies.jar
└── Log/                          # Log files directory
```

## Main Functional Modules

### 1. Permission Extraction
- **Location**: `src/extract_permission/`
- **Function**: Extract application permission information from Android APK files
- **Output**: Permission list files

### 2. Call Graph Extraction
- **Location**: `src/extract_call_graph/`
- **Function**: Analyze APK code structure and generate method call relationship graphs
- **Tools**: Use Soot framework for static analysis
- **Output**: Call graph files

### 3. Taint Path Extraction
- **Location**: `src/extract_taint_path/`
- **Function**: Perform taint analysis and track data flow paths
- **Tools**: Use Soot Infoflow for taint analysis
- **Output**: Taint path XML files

### 4. Abstract Semantic Graph Generation
- **Location**: `src/absGraph/`
- **Function**: Integrate permission, call graph, and taint path information to generate abstract semantic graphs
- **Output**: Graph feature files for machine learning model training

## Usage Workflow

### Extract Permission

python extract_permission.py -d ../../Dataset/2022/malware

### Extract Call Graph

python extract_call_graphs.py -d ../../Dataset/2022/malware/ -pd ../../Lib/platforms/

### Extract Taint Path

python taint_analysis.py -d ../../Dataset/2022/malware/

python parseXML.py -tf ../../Raw/2022/taint_path_malware_taint_path_xml/

### ASG Generation

python generate_absGraph.py --call_graph_dir ../../Raw/2022/call_graph/malware/graphs --taint_path_dir ../../Raw/2022/taint_path/malware/taint_path_txt/ --permission_dir ../../Raw/2022/permission/malware/permissions/ --output_dir ../../Features/2022/malware --label 1 --codebert_embeddings_path ../nodeRepresentation/codebert_api_embeddings.npz --timeout 60 