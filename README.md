# Socioeconomic Factors affecting Healthcare in America
Team Project that examines and illustrates how socioeconomic factors affect patient outcomes in America. 

## Introduction

Quality of care and patient outcomes are significant determinants of the effectiveness of a healthcare system. My team’s project expands upon the best practices in healthcare analytics for evaluating the current status of patient care, and analyzes relationships between various factors that drive cost and influence high quality outcomes. Our project summarizes the findings in a comprehensive, interactive representation of these factors to provide enhanced insight into the main drivers of quality of healthcare, and ultimately facilitates improved patient control and targeted improvement efforts.

## Getting Started

Select your destination folder for the patient socioeconomic and outcome table. Both the R and Python scripts are set to run with relative paths, so placing both of these files into the same destination folder is enough (no need to set a working directory). 

### Prerequisites

* Python IDE
* R IDE

```
e.g. R Studio, Spyder, Jupyter
```
<!---

### Run

1. To run the pagerank.py algorithm, follow the steps below. 
1. Since memory mapping works with binary files, the graph’s edge list needs to be converted into its binary format by running the following command at the terminal/command prompt (you only need to do this once):

    ```python q1_utils.py convert <path-to-edgelist.txt>```

    This generates 3 files:
    * A .bin binary file containing edges (source, target) in big-endian “int” C type
    * A .idx: binary file containing (node, degree) in little-endian “int” C type
    * A .json: metadata about the conversion process (required to run pagerank)

2. To execute the PageRank algorithm, type the following code into the command line/terminal:

    ```pypy q1_utils.py pagerank <path to JSON file for LiveJournal>```

    This will output the 10 nodes witht he highest PageRank scores. The default number of iterations is 10. The number of iterations can be updated by adding the desired number to the end of the command:
    ```pypy q1_utils.py pagerank toy-graph/toy-graph.json --iterations 25``
    A file in the format pagerank_nodes_n.txt  for “n” number of iterations will be created.

### Acknowledgments
