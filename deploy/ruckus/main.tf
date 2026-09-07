terraform {
  required_version = ">= 1.10, < 2.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "8.1.0"
    }
  }
}

provider "google" {
  project = "cohesive-mark-500400-k3"
  region  = "us-central1"
  zone    = "us-central1-a"
}

# A custom external OAuth client is required because the designated admin may
# be outside the Google organization that owns this project. No fallback to a
# Google-managed client: that can lock the administrator out.
variable "iap_oauth_client_id" {
  type        = string
  description = "External Google OAuth web client configured for IAP"
  validation {
    condition     = endswith(var.iap_oauth_client_id, ".apps.googleusercontent.com")
    error_message = "Supply the external OAuth client ID before planning."
  }
}

variable "iap_oauth_client_secret" {
  type        = string
  sensitive   = true
  description = "External OAuth client secret; protect the local state file"
  validation {
    condition     = length(var.iap_oauth_client_secret) > 0
    error_message = "An OAuth client secret is required."
  }
}

locals {
  project     = "cohesive-mark-500400-k3"
  admin_email = "jake@sixtyoneeighty.com"
  hostname    = "ruckus.sixtyoneeighty.com"
}

# Apply is a separate, explicit operation. Never disable project-wide APIs
# when removing this deployment.
resource "google_project_service" "required" {
  for_each = toset([
    "compute.googleapis.com",
    "iap.googleapis.com",
    "certificatemanager.googleapis.com",
  ])
  project            = local.project
  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "ruckus" {
  name                    = "ruckus-private"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.required]
}

resource "google_compute_subnetwork" "ruckus" {
  name          = "ruckus-private-central"
  ip_cidr_range = "10.68.0.0/24"
  region        = "us-central1"
  network       = google_compute_network.ruckus.id
}

resource "google_compute_firewall" "gateway" {
  name          = "ruckus-load-balancer-only"
  network       = google_compute_network.ruckus.name
  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
  target_tags   = ["ruckus-private"]
  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }
}

resource "google_compute_firewall" "ssh" {
  name          = "ruckus-iap-ssh-only"
  network       = google_compute_network.ruckus.name
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["ruckus-private"]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

# No project roles or OAuth scopes. Do not give agent code the default
# Compute Engine service account's project permissions.
resource "google_service_account" "runtime" {
  account_id   = "ruckus-runtime"
  display_name = "Ruckus runtime - no project permissions"
}

resource "google_compute_disk" "state" {
  name = "ruckus-state"
  type = "pd-balanced"
  zone = "us-central1-a"
  size = 30
  lifecycle {
    prevent_destroy = true
  }
  depends_on = [google_project_service.required]
}

resource "google_compute_instance" "ruckus" {
  name                = "ruckus-private"
  machine_type        = "e2-standard-2"
  zone                = "us-central1-a"
  deletion_protection = true
  tags                = ["ruckus-private"]
  labels = {
    application = "ruckus"
    stage       = "private-staging"
  }
  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = 20
      type  = "pd-balanced"
    }
  }
  attached_disk {
    source      = google_compute_disk.state.id
    device_name = "ruckus-state"
  }
  network_interface {
    subnetwork = google_compute_subnetwork.ruckus.id
    # Egress without Cloud NAT. Ingress is restricted to Google LB and IAP
    # source ranges above; no agent-server port is publicly reachable.
    access_config {}
  }
  service_account {
    email  = google_service_account.runtime.email
    scopes = []
  }
  metadata = {
    enable-oslogin         = "TRUE"
    block-project-ssh-keys = "TRUE"
  }
  metadata_startup_script = file("${path.module}/startup.sh")
  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }
}

resource "google_compute_instance_group" "ruckus" {
  name      = "ruckus-private"
  zone      = "us-central1-a"
  instances = [google_compute_instance.ruckus.self_link]
  named_port {
    name = "gateway"
    port = 8080
  }
}

resource "google_compute_health_check" "gateway" {
  name = "ruckus-gateway"
  http_health_check {
    port         = 8080
    request_path = "/_health"
  }
  depends_on = [google_project_service.required]
}

resource "google_compute_backend_service" "ruckus" {
  name                  = "ruckus-private"
  protocol              = "HTTP"
  port_name             = "gateway"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  timeout_sec           = 3600
  enable_cdn            = false
  health_checks         = [google_compute_health_check.gateway.id]
  backend {
    group = google_compute_instance_group.ruckus.id
  }
  iap {
    enabled              = true
    oauth2_client_id     = var.iap_oauth_client_id
    oauth2_client_secret = var.iap_oauth_client_secret
  }
}

resource "google_iap_web_backend_service_iam_member" "admin" {
  project             = local.project
  web_backend_service = google_compute_backend_service.ruckus.name
  role                = "roles/iap.httpsResourceAccessor"
  member              = "user:${local.admin_email}"
}

# DNS authorization can issue the certificate while the live app stays on
# Vercel. Publishing this unique validation CNAME is NOT the domain cutover.
resource "google_certificate_manager_dns_authorization" "ruckus" {
  name       = "ruckus-domain"
  domain     = local.hostname
  type       = "PER_PROJECT_RECORD"
  depends_on = [google_project_service.required]
}

resource "google_certificate_manager_certificate" "ruckus" {
  name = "ruckus-domain"
  managed {
    domains            = [local.hostname]
    dns_authorizations = [google_certificate_manager_dns_authorization.ruckus.id]
  }
}

resource "google_certificate_manager_certificate_map" "ruckus" {
  name       = "ruckus-certificates"
  depends_on = [google_project_service.required]
}

resource "google_certificate_manager_certificate_map_entry" "ruckus" {
  name         = "ruckus-domain"
  map          = google_certificate_manager_certificate_map.ruckus.name
  certificates = [google_certificate_manager_certificate.ruckus.id]
  hostname     = local.hostname
}

resource "google_compute_url_map" "ruckus" {
  name            = "ruckus-private"
  default_service = google_compute_backend_service.ruckus.id
}

resource "google_compute_target_https_proxy" "ruckus" {
  name            = "ruckus-private"
  url_map         = google_compute_url_map.ruckus.id
  certificate_map = "//certificatemanager.googleapis.com/${google_certificate_manager_certificate_map.ruckus.id}"
}

resource "google_compute_global_address" "ruckus" {
  name       = "ruckus-private"
  depends_on = [google_project_service.required]
}

resource "google_compute_global_forwarding_rule" "https" {
  name                  = "ruckus-private-https"
  target                = google_compute_target_https_proxy.ruckus.id
  ip_address            = google_compute_global_address.ruckus.address
  port_range            = "443"
  load_balancing_scheme = "EXTERNAL_MANAGED"
}

output "dns_validation" {
  value = google_certificate_manager_dns_authorization.ruckus.dns_resource_record
}

output "future_dns_cutover_address" {
  value = google_compute_global_address.ruckus.address
}

output "gateway_environment" {
  value = {
    RUCKUS_ADMIN_EMAIL  = local.admin_email
    RUCKUS_PUBLIC_HOST  = local.hostname
    RUCKUS_IAP_AUDIENCE = "/projects/799294352663/global/backendServices/${google_compute_backend_service.ruckus.generated_id}"
  }
}
