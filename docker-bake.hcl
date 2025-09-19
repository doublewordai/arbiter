# Docker Bake file for building arbiter images with multiplatform support and attestations

variable "REGISTRY" {
  default = "ghcr.io/doublewordai/arbiter"
}

# Build platforms
variable "PLATFORMS" {
  default = "linux/amd64,linux/arm64"
}

# Global tags (comma-separated)
variable "TAGS" {
  default = ""
}

# CPU inference server
target "cpu" {
  context = "."
  dockerfile = "Dockerfile"
  target = "cpu"
  tags = TAGS != "" ? [for tag in split(",", TAGS) : "${REGISTRY}:${tag}"] : []
  labels = {}
  platforms = split(",", PLATFORMS)
  annotations = []
  attest = [
    "type=provenance,mode=max",
    "type=sbom"
  ]
}

# GPU inference server
target "gpu" {
  context = "."
  dockerfile = "Dockerfile"
  target = "gpu"
  tags = TAGS != "" ? [for tag in split(",", TAGS) : "${REGISTRY}:${tag}-gpu"] : []
  labels = {}
  platforms = split(",", PLATFORMS)
  annotations = []
  attest = [
    "type=provenance,mode=max",
    "type=sbom"
  ]
}

# Group target for building all images
group "default" {
  targets = ["cpu", "gpu"]
}