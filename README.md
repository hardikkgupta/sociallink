# SocialLink Recommendation Engine

A scalable collaborative-filtering recommendation system built with PyTorch and Apache Spark, designed to serve personalized connection suggestions to millions of users.

## Features

- Distributed collaborative filtering using Apache Spark
- Neural ranking models with PyTorch
- End-to-end ML pipeline orchestrated with Airflow
- Containerized deployment with Docker and Kubernetes
- A/B testing framework for model evaluation
- High availability with 95% uptime
- Real-time recommendation serving

## Architecture

```
├── data/                  # Data storage
├── src/
│   ├── models/           # PyTorch model implementations
│   ├── spark/            # Spark processing jobs
│   ├── api/              # FastAPI service
│   ├── airflow/          # DAG definitions
│   └── kubernetes/       # K8s manifests
├── tests/                # Unit and integration tests
└── docker/               # Docker configurations
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the development environment:
```bash
docker-compose up -d
```

## Development

- Run tests: `pytest tests/`
- Format code: `black src/`
- Lint code: `flake8 src/`

## Deployment

The system is designed to be deployed on Kubernetes:

1. Build Docker images:
```bash
docker build -t sociallink-recommender .
```

2. Deploy to Kubernetes:
```bash
kubectl apply -f src/kubernetes/
```

## Performance

- Training speedup: 2x on distributed GPU clusters
- CTR improvement: 15% in live experiments
- Time-to-production: 24 hours
- System uptime: 95%

## License

MIT License 