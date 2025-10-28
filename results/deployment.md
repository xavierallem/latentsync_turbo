# LatentSync AWS Deployment Framework

This deployment guide is designed for AWS infrastructure, as I heard the team is planning to migrate to AWS soon. 

## Architecture

```
Users → API Gateway → ALB → EKS (GPU Pods) → S3
                                ↓
                          Prometheus → Grafana
```

The architecture follows a microservices pattern with asynchronous job processing:

- **API Gateway**: Accepts video processing requests and returns job status
- **Application Load Balancer (ALB)**: Distributes traffic to Kubernetes pods
- **EKS Cluster**: Manages GPU-enabled pods running LatentSync inference
- **S3**: Stores input videos, processed outputs, and model weights
- **SQS**: Queues jobs for reliable asynchronous processing
- **Prometheus/Grafana**: Monitors system performance and health

---

## Infrastructure Components

### AWS Services

**Compute:**
- EKS cluster (Kubernetes 1.27+)
- GPU nodes: g5.xlarge instances (24GB VRAM)
- NVIDIA device plugin for GPU support
- Auto-scaling: 1-10 nodes

**Storage:**
- S3 input bucket (videos/audio)
- S3 output bucket (processed videos)
- S3 models bucket (weights ~5-10GB)
- EBS volumes (100GB per pod)

**Network:**
- VPC with public/private subnets (2+ AZs)
- Application Load Balancer (HTTPS)
- NAT Gateway for outbound access
- VPC endpoints for S3

---

## Kubernetes Setup

### Pod Configuration

**Resources per Pod:**
- GPU: 1 (exclusive)
- Memory: 16GB
- CPU: 4 cores
- Storage: 100GB
- Processing time: 90-180s per video [Comes inline with the efficient and will drastically improve with a good gpu]

**Replicas:**
- Min: 1, Max: 10
- Auto-scale on queue depth
- One video per pod

### Auto-Scaling

- Scale up when queue >5 per pod
- Scale down when queue <2 per pod
- Cluster auto-scaler adds/removes nodes
- 5-minute cooldown between scaling

---

## Application Architecture

### API Endpoints

- `POST /process` - Submit job
- `GET /status/{id}` - Job status
- `GET /result/{id}` - Download result
- `GET /metrics` - Prometheus metrics

### Job Processing

**Flow:**
1. Upload → S3 input bucket
2. Queue job in SQS
3. Pod processes video
4. Upload → S3 output bucket
5. Delete from queue

**Queue (SQS):**
- FIFO for ordering
- 300s visibility timeout
- Dead letter queue for failures
- Max 3 retries

---

## Monitoring Setup

### Prometheus

**Deploy:**
- Install via Helm in monitoring namespace
- Scrape interval: 30s
- 15-day retention
- 100GB storage

**Key Metrics:**
- Videos processed (counter)
- Queue size (gauge)
- Active jobs (gauge)
- Processing time (histogram)
- GPU utilization (gauge)
- Failed jobs (counter)
- API requests (counter)

### Grafana

**Deploy:**
- Install via Helm
- Connect to Prometheus data source
- HTTPS access via load balancer

**Dashboards:**

1. **Overview:**
   - Videos processed today
   - Queue size
   - Processing time avg
   - Throughput (videos/hour)

2. **Resources:**
   - GPU utilization per pod
   - Memory/CPU usage
   - Queue depth over time

3. **Errors:**
   - Failed jobs timeline
   - Error type distribution
   - Retry success rates

**Alerts:**
- High error rate (>10% for 5min)
- Long processing time (>300s)
- Large queue (>50 jobs)
- Pods down

---

## Deployment Steps

### 1. Prepare Infrastructure
- Create EKS cluster with GPU node group
- Configure VPC, subnets, security groups
- Set up S3 buckets
- Upload model weights to S3

### 2. Configure Kubernetes
- Install NVIDIA device plugin
- Deploy LatentSync application [FastAPI Backend]
- Configure service and load balancer
- Set up horizontal pod autoscaler

### 3. Set Up Monitoring
- Install Prometheus via Helm
- Install Grafana via Helm
- Configure Prometheus to scrape pods
- Create Grafana dashboards
- Set up alert rules

### 4. Configure Queue
- Create SQS queue (FIFO)
- Set up dead letter queue
- Configure visibility timeout
- Connect to application

### 5. Test & Validate
- Submit test video
- Monitor processing in Grafana
- Verify auto-scaling behavior
- Test failure scenarios

---

## Key Configurations

### Environment Variables (Pods)
- S3_INPUT_BUCKET
- S3_OUTPUT_BUCKET
- S3_MODELS_BUCKET
- AWS_REGION
- INFERENCE_STEPS (20-50)
- GUIDANCE_SCALE (1.0-3.0)


## Token/Usage Tracking

**Input Metrics:**
- Video size (MB)
- Video duration (seconds)
- Resolution
- Frame count

**Output Metrics:**
- Processing time (seconds)
- GPU hours consumed
- Generated file size
- Inference steps used

**Per-Tenant Tracking:**
- Total videos processed
- Total GPU time
- Storage used
- Aggregate daily/monthly

---

## Summary

**Complete Stack:**
- AWS EKS for orchestration
- GPU instances for processing
- S3 for storage
- SQS for job queue
- Prometheus for metrics
- Grafana for visualization

**Scaling Strategy:**
- Auto-scale pods based on queue
- Auto-scale nodes based on pending pods
- Min 1 replica, max 10
- Process 20-40 videos/hour per GPU

**Monitoring:**
- Real-time metrics in Grafana
- Alerts for critical issues
- Track usage per user/tenant
- Performance optimization data