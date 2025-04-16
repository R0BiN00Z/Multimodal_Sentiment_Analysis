# Multimodal Sentiment Analysis

This project focuses on multimodal sentiment analysis using the CMU-MOSEI dataset.

## Project Status

- [x] Project structure setup
- [x] CMU-MOSEI dataset download script
- [x] Feature alignment implementation
- [ ] Model development
- [ ] Training and evaluation
- [ ] Results analysis

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd multimodal_sentiment_analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download CMU-MOSEI dataset:
```bash
python scripts/download_cmu_mosei.py
```

## Project Structure

```
multimodal_sentiment_analysis/
├── data/                    # Dataset storage
│   └── CMU_MOSEI/          # CMU-MOSEI dataset files
├── scripts/                 # Utility scripts
│   ├── download_cmu_mosei.py  # Dataset download script
│   └── align_mosei.py      # Feature alignment script
├── models/                  # Model implementations
├── utils/                   # Utility functions
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Current Progress

- Successfully implemented dataset download script
- Working on feature alignment using 'All Labels' as alignment key
- Dataset includes:
  - GloVe word vectors
  - COVAREP acoustic features
  - OpenFace visual features
  - FACET 4.2 visual features
  - Sentiment labels

## Next Steps

1. Complete feature alignment process
2. Implement baseline models
3. Develop multimodal fusion approaches
4. Train and evaluate models
5. Analyze results and optimize performance

## Requirements

- Python 3.x
- mmsdk
- numpy
- torch
- Other dependencies listed in requirements.txt

## Contributing

Feel free to submit issues and enhancement requests.

## License

[Add your license information here]
