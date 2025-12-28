# OpenHands Enterprise Usage Telemetry Service

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [Problem Statement](#11-problem-statement)
   - 1.2 [Proposed Solution](#12-proposed-solution)
2. [User Interface](#2-user-interface)
   - 2.1 [License Warning Banner](#21-license-warning-banner)
   - 2.2 [Administrator Experience](#22-administrator-experience)
3. [Other Context](#3-other-context)
   - 3.1 [Replicated Platform Integration](#31-replicated-platform-integration)
   - 3.2 [Administrator Email Detection Strategy](#32-administrator-email-detection-strategy)
   - 3.3 [Metrics Collection Framework](#33-metrics-collection-framework)
4. [Technical Design](#4-technical-design)
   - 4.1 [Database Schema](#41-database-schema)
     - 4.1.1 [Telemetry Metrics Table](#411-telemetry-metrics-table)
     - 4.1.2 [Telemetry Identity Table](#412-telemetry-identity-table)
   - 4.2 [Metrics Collection Framework](#42-metrics-collection-framework)
     - 4.2.1 [Base Collector Interface](#421-base-collector-interface)
     - 4.2.2 [Collector Registry](#422-collector-registry)
     - 4.2.3 [Example Collector Implementation](#423-example-collector-implementation)
   - 4.3 [Collection and Upload System](#43-collection-and-upload-system)
     - 4.3.1 [Metrics Collection Processor](#431-metrics-collection-processor)
     - 4.3.2 [Replicated Upload Processor](#432-replicated-upload-processor)
   - 4.4 [License Warning System](#44-license-warning-system)
     - 4.4.1 [License Status Endpoint](#441-license-status-endpoint)
     - 4.4.2 [UI Integration](#442-ui-integration)
   - 4.5 [Cronjob Configuration](#45-cronjob-configuration)
     - 4.5.1 [Collection Cronjob](#451-collection-cronjob)
     - 4.5.2 [Upload Cronjob](#452-upload-cronjob)
5. [Implementation Plan](#5-implementation-plan)
   - 5.1 [Database Schema and Models (M1)](#51-database-schema-and-models-m1)
     - 5.1.1 [OpenHands - Database Migration](#511-openhands---database-migration)
     - 5.1.2 [OpenHands - Model Tests](#512-openhands---model-tests)
   - 5.2 [Metrics Collection Framework (M2)](#52-metrics-collection-framework-m2)
     - 5.2.1 [OpenHands - Core Collection Framework](#521-openhands---core-collection-framework)
     - 5.2.2 [OpenHands - Example Collectors](#522-openhands---example-collectors)
     - 5.2.3 [OpenHands - Framework Tests](#523-openhands---framework-tests)
   - 5.3 [Collection and Upload Processors (M3)](#53-collection-and-upload-processors-m3)
     - 5.3.1 [OpenHands - Collection Processor](#531-openhands---collection-processor)
     - 5.3.2 [OpenHands - Upload Processor](#532-openhands---upload-processor)
     - 5.3.3 [OpenHands - Integration Tests](#533-openhands---integration-tests)
   - 5.4 [License Warning API (M4)](#54-license-warning-api-m4)
     - 5.4.1 [OpenHands - License Status API](#541-openhands---license-status-api)
     - 5.4.2 [OpenHands - API Integration](#542-openhands---api-integration)
   - 5.5 [UI Warning Banner (M5)](#55-ui-warning-banner-m5)
     - 5.5.1 [OpenHands - UI Warning Banner](#551-openhands---ui-warning-banner)
     - 5.5.2 [OpenHands - UI Integration](#552-openhands---ui-integration)
   - 5.6 [Helm Chart Deployment Configuration (M6)](#56-helm-chart-deployment-configuration-m6)
     - 5.6.1 [OpenHands-Cloud - Cronjob Manifests](#561-openhands-cloud---cronjob-manifests)
     - 5.6.2 [OpenHands-Cloud - Configuration Management](#562-openhands-cloud---configuration-management)
   - 5.7 [Documentation and Enhanced Collectors (M7)](#57-documentation-and-enhanced-collectors-m7)
     - 5.7.1 [OpenHands - Advanced Collectors](#571-openhands---advanced-collectors)
     - 5.7.2 [OpenHands - Monitoring and Testing](#572-openhands---monitoring-and-testing)
     - 5.7.3 [OpenHands - Technical Documentation](#573-openhands---technical-documentation)

## 1. Introduction

### 1.1 Problem Statement

OpenHands Enterprise (OHE) helm charts are publicly available but not open source, creating a visibility gap for the sales team. Unknown users can install and use OHE without the vendor's knowledge, preventing proper customer engagement and sales pipeline management. Without usage telemetry, the vendor cannot identify potential customers, track installation health, or proactively support users who may need assistance.

### 1.2 Proposed Solution

We propose implementing a comprehensive telemetry service that leverages the Replicated metrics platform and Python SDK to track OHE installations and usage. The solution provides automatic customer discovery, instance monitoring, and usage metrics collection while maintaining a clear license compliance pathway.

The system consists of three main components: (1) a pluggable metrics collection framework that allows developers to easily define and register custom metrics collectors, (2) automated cronjobs that periodically collect metrics and upload them to Replicated's vendor portal, and (3) a license compliance warning system that displays UI notifications when telemetry uploads fail, indicating potential license expiration.

The design ensures that telemetry cannot be easily disabled without breaking core OHE functionality by tying the warning system to environment variables that are essential for OHE operation. This approach balances user transparency with business requirements for customer visibility.

## 2. User Interface

### 2.1 License Warning Banner

When telemetry uploads fail for more than 4 days, users will see a prominent warning banner in the OpenHands Enterprise UI:

```
⚠️ Your OpenHands Enterprise license will expire in 30 days. Please contact support if this issue persists.
```

The banner appears at the top of all pages and cannot be permanently dismissed while the condition persists. Users can temporarily dismiss it, but it will reappear on page refresh until telemetry uploads resume successfully.

### 2.2 Administrator Experience

System administrators will not need to configure the telemetry system manually. The service automatically:

1. **Detects OHE installations** using existing required environment variables (`GITHUB_APP_CLIENT_ID`, `KEYCLOAK_SERVER_URL`, etc.)

2. **Generates unique customer identifiers** using administrator contact information:
   - Customer email: Determined by the following priority order:
     1. `OPENHANDS_ADMIN_EMAIL` environment variable (if set in helm values)
     2. Email of the first user who accepted Terms of Service (earliest `accepted_tos` timestamp)
   - Instance ID: Automatically generated by Replicated SDK using machine fingerprinting (IOPlatformUUID on macOS, D-Bus machine ID on Linux, Machine GUID on Windows)
   - **No Fallback**: If neither email source is available, telemetry collection is skipped until at least one user exists

3. **Collects and uploads metrics transparently** in the background via weekly collection and daily upload cronjobs

4. **Displays warnings only when necessary** for license compliance - no notifications appear during normal operation

## 3. Other Context

### 3.1 Replicated Platform Integration

The Replicated platform provides vendor-hosted infrastructure for collecting customer and instance telemetry. The Python SDK handles authentication, state management, and reliable metric delivery. Key concepts:

- **Customer**: Represents a unique OHE installation, identified by email or installation fingerprint
- **Instance**: Represents a specific deployment of OHE for a customer
- **Metrics**: Custom key-value data points collected from the installation
- **Status**: Instance health indicators (running, degraded, updating, etc.)

The SDK automatically handles machine fingerprinting, local state caching, and retry logic for failed uploads.

### 3.2 Administrator Email Detection Strategy

To identify the appropriate administrator contact for sales outreach, the system uses a three-tier approach that avoids performance penalties on user authentication:

**Tier 1: Explicit Configuration** - The `OPENHANDS_ADMIN_EMAIL` environment variable allows administrators to explicitly specify the contact email during deployment.

**Tier 2: First Active User Detection** - If no explicit email is configured, the system identifies the first user who accepted Terms of Service (earliest `accepted_tos` timestamp with a valid email). This represents the first person to actively engage with the system and is very likely the administrator or installer.

**No Fallback Needed** - If neither email source is available, telemetry collection is skipped entirely. This ensures we only report meaningful usage data when there are actual active users.

**Performance Optimization**: The admin email determination is performed only during telemetry upload attempts, ensuring zero performance impact on user login flows.

### 3.3 Metrics Collection Framework

The proposed collector framework allows developers to define metrics in a single file change:

```python
@register_collector("user_activity")
class UserActivityCollector(MetricsCollector):
    def collect(self) -> Dict[str, Any]:
        # Query database and return metrics
        return {"active_users_7d": count, "conversations_created": total}
```

Collectors are automatically discovered and executed by the collection cronjob, making the system extensible without modifying core collection logic.

## 4. Technical Design

### 4.1 Database Schema

#### 4.1.1 Telemetry Metrics Table

Stores collected metrics with transmission status tracking:

```sql
CREATE TABLE telemetry_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metrics_data JSONB NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE NULL,
    upload_attempts INTEGER DEFAULT 0,
    last_upload_error TEXT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_telemetry_metrics_collected_at ON telemetry_metrics(collected_at);
CREATE INDEX idx_telemetry_metrics_uploaded_at ON telemetry_metrics(uploaded_at);
```

#### 4.1.2 Telemetry Identity Table

Stores persistent identity information that must survive container restarts:

```sql
CREATE TABLE telemetry_identity (
    id INTEGER PRIMARY KEY DEFAULT 1,
    customer_id VARCHAR(255) NULL,
    instance_id VARCHAR(255) NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT single_identity_row CHECK (id = 1)
);
```

**Design Rationale:**
- **Separation of Concerns**: Identity data (customer_id, instance_id) is separated from operational data
- **Persistent vs Computed**: Only data that cannot be reliably recomputed is persisted
- **Upload Tracking**: Upload timestamps are tied directly to the metrics they represent
- **Simplified Queries**: System state can be derived from metrics table (e.g., `MAX(uploaded_at)` for last successful upload)

### 4.2 Metrics Collection Framework

#### 4.2.1 Base Collector Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class MetricResult:
    key: str
    value: Any

class MetricsCollector(ABC):
    """Base class for metrics collectors."""

    @abstractmethod
    def collect(self) -> List[MetricResult]:
        """Collect metrics and return results."""
        pass

    @property
    @abstractmethod
    def collector_name(self) -> str:
        """Unique name for this collector."""
        pass

    def should_collect(self) -> bool:
        """Override to add collection conditions."""
        return True
```

#### 4.2.2 Collector Registry

```python
from typing import Dict, Type, List
import importlib
import pkgutil

class CollectorRegistry:
    """Registry for metrics collectors."""

    def __init__(self):
        self._collectors: Dict[str, Type[MetricsCollector]] = {}

    def register(self, collector_class: Type[MetricsCollector]) -> None:
        """Register a collector class."""
        collector = collector_class()
        self._collectors[collector.collector_name] = collector_class

    def get_all_collectors(self) -> List[MetricsCollector]:
        """Get instances of all registered collectors."""
        return [cls() for cls in self._collectors.values()]

    def discover_collectors(self, package_path: str) -> None:
        """Auto-discover collectors in a package."""
        # Implementation to scan for @register_collector decorators
        pass

# Global registry instance
collector_registry = CollectorRegistry()

def register_collector(name: str):
    """Decorator to register a collector."""
    def decorator(cls: Type[MetricsCollector]) -> Type[MetricsCollector]:
        collector_registry.register(cls)
        return cls
    return decorator
```

#### 4.2.3 Example Collector Implementation

```python
@register_collector("system_metrics")
class SystemMetricsCollector(MetricsCollector):
    """Collects basic system and usage metrics."""

    @property
    def collector_name(self) -> str:
        return "system_metrics"

    def collect(self) -> List[MetricResult]:
        results = []

        # Collect user count
        with session_maker() as session:
            user_count = session.query(UserSettings).count()
            results.append(MetricResult(
                key="total_users",
                value=user_count
            ))

            # Collect conversation count (last 30 days)
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            conversation_count = session.query(StoredConversationMetadata)\
                .filter(StoredConversationMetadata.created_at >= thirty_days_ago)\
                .count()

            results.append(MetricResult(
                key="conversations_30d",
                value=conversation_count
            ))

        return results
```

### 4.3 Collection and Upload System

#### 4.3.1 Metrics Collection Processor

```python
class TelemetryCollectionProcessor(MaintenanceTaskProcessor):
    """Maintenance task processor for collecting metrics."""

    collection_interval_days: int = 7

    async def __call__(self, task: MaintenanceTask) -> dict:
        """Collect metrics from all registered collectors."""

        # Check if collection is needed
        if not self._should_collect():
            return {"status": "skipped", "reason": "too_recent"}

        # Collect metrics from all registered collectors
        all_metrics = {}
        collector_results = {}

        for collector in collector_registry.get_all_collectors():
            try:
                if collector.should_collect():
                    results = collector.collect()
                    for result in results:
                        all_metrics[result.key] = result.value
                    collector_results[collector.collector_name] = len(results)
            except Exception as e:
                logger.error(f"Collector {collector.collector_name} failed: {e}")
                collector_results[collector.collector_name] = f"error: {e}"

        # Store metrics in database
        with session_maker() as session:
            telemetry_record = TelemetryMetrics(
                metrics_data=all_metrics,
                collected_at=datetime.now(timezone.utc)
            )
            session.add(telemetry_record)
            session.commit()

            # Note: No need to track last_collection_at separately
            # Can be derived from MAX(collected_at) in telemetry_metrics

        return {
            "status": "completed",
            "metrics_collected": len(all_metrics),
            "collectors_run": collector_results
        }

    def _should_collect(self) -> bool:
        """Check if collection is needed based on interval."""
        with session_maker() as session:
            # Get last collection time from metrics table
            last_collected = session.query(func.max(TelemetryMetrics.collected_at)).scalar()
            if not last_collected:
                return True

            time_since_last = datetime.now(timezone.utc) - last_collected
            return time_since_last.days >= self.collection_interval_days
```

#### 4.3.2 Replicated Upload Processor

```python
from replicated import AsyncReplicatedClient, InstanceStatus

class TelemetryUploadProcessor(MaintenanceTaskProcessor):
    """Maintenance task processor for uploading metrics to Replicated."""

    replicated_publishable_key: str
    replicated_app_slug: str

    async def __call__(self, task: MaintenanceTask) -> dict:
        """Upload pending metrics to Replicated."""

        # Get pending metrics
        with session_maker() as session:
            pending_metrics = session.query(TelemetryMetrics)\
                .filter(TelemetryMetrics.uploaded_at.is_(None))\
                .order_by(TelemetryMetrics.collected_at)\
                .all()

        if not pending_metrics:
            return {"status": "no_pending_metrics"}

        # Get admin email - skip if not available
        admin_email = self._get_admin_email()
        if not admin_email:
            logger.info("Skipping telemetry upload - no admin email available")
            return {
                "status": "skipped",
                "reason": "no_admin_email",
                "total_processed": 0
            }

        uploaded_count = 0
        failed_count = 0

        async with AsyncReplicatedClient(
            publishable_key=self.replicated_publishable_key,
            app_slug=self.replicated_app_slug
        ) as client:

            # Get or create customer and instance
            customer = await client.customer.get_or_create(
                email_address=admin_email
            )
            instance = await customer.get_or_create_instance()

            # Store customer/instance IDs for future use
            await self._update_telemetry_identity(customer.customer_id, instance.instance_id)

            # Upload each metric batch
            for metric_record in pending_metrics:
                try:
                    # Send individual metrics
                    for key, value in metric_record.metrics_data.items():
                        await instance.send_metric(key, value)

                    # Update instance status
                    await instance.set_status(InstanceStatus.RUNNING)

                    # Mark as uploaded
                    with session_maker() as session:
                        record = session.query(TelemetryMetrics)\
                            .filter(TelemetryMetrics.id == metric_record.id)\
                            .first()
                        if record:
                            record.uploaded_at = datetime.now(timezone.utc)
                            session.commit()

                    uploaded_count += 1

                except Exception as e:
                    logger.error(f"Failed to upload metrics {metric_record.id}: {e}")

                    # Update error info
                    with session_maker() as session:
                        record = session.query(TelemetryMetrics)\
                            .filter(TelemetryMetrics.id == metric_record.id)\
                            .first()
                        if record:
                            record.upload_attempts += 1
                            record.last_upload_error = str(e)
                            session.commit()

                    failed_count += 1

        # Note: No need to track last_successful_upload_at separately
        # Can be derived from MAX(uploaded_at) in telemetry_metrics

        return {
            "status": "completed",
            "uploaded": uploaded_count,
            "failed": failed_count,
            "total_processed": len(pending_metrics)
        }

    def _get_admin_email(self) -> str | None:
        """Get administrator email for customer identification."""
        # 1. Check environment variable first
        env_admin_email = os.getenv('OPENHANDS_ADMIN_EMAIL')
        if env_admin_email:
            logger.info("Using admin email from environment variable")
            return env_admin_email

        # 2. Use first active user's email (earliest accepted_tos)
        with session_maker() as session:
            first_user = session.query(UserSettings)\
                .filter(UserSettings.email.isnot(None))\
                .filter(UserSettings.accepted_tos.isnot(None))\
                .order_by(UserSettings.accepted_tos.asc())\
                .first()

            if first_user and first_user.email:
                logger.info(f"Using first active user email: {first_user.email}")
                return first_user.email

        # No admin email available - skip telemetry
        logger.info("No admin email available - skipping telemetry collection")
        return None

    async def _update_telemetry_identity(self, customer_id: str, instance_id: str) -> None:
        """Update or create telemetry identity record."""
        with session_maker() as session:
            identity = session.query(TelemetryIdentity).first()
            if not identity:
                identity = TelemetryIdentity()
                session.add(identity)

            identity.customer_id = customer_id
            identity.instance_id = instance_id
            session.commit()
```

### 4.4 License Warning System

#### 4.4.1 License Status Endpoint

```python
from fastapi import APIRouter
from datetime import datetime, timezone, timedelta

license_router = APIRouter()

@license_router.get("/license-status")
async def get_license_status():
    """Get license warning status for UI display."""

    # Only show warnings for OHE installations
    if not _is_openhands_enterprise():
        return {"warn": False, "message": ""}

    with session_maker() as session:
        # Get last successful upload time from metrics table
        last_upload = session.query(func.max(TelemetryMetrics.uploaded_at))\
            .filter(TelemetryMetrics.uploaded_at.isnot(None))\
            .scalar()

        if not last_upload:
            # No successful uploads yet - show warning after 4 days
            return {
                "warn": True,
                "message": "OpenHands Enterprise license verification pending. Please ensure network connectivity."
            }

        # Check if last successful upload was more than 4 days ago
        days_since_upload = (datetime.now(timezone.utc) - last_upload).days

        if days_since_upload > 4:
            # Find oldest unsent batch
            oldest_unsent = session.query(TelemetryMetrics)\
                .filter(TelemetryMetrics.uploaded_at.is_(None))\
                .order_by(TelemetryMetrics.collected_at)\
                .first()

            if oldest_unsent:
                # Calculate expiration date (oldest unsent + 34 days)
                expiration_date = oldest_unsent.collected_at + timedelta(days=34)
                days_until_expiration = (expiration_date - datetime.now(timezone.utc)).days

                if days_until_expiration <= 0:
                    message = "Your OpenHands Enterprise license has expired. Please contact support immediately."
                else:
                    message = f"Your OpenHands Enterprise license will expire in {days_until_expiration} days. Please contact support if this issue persists."

                return {"warn": True, "message": message}

        return {"warn": False, "message": ""}

def _is_openhands_enterprise() -> bool:
    """Detect if this is an OHE installation."""
    # Check for required OHE environment variables
    required_vars = [
        'GITHUB_APP_CLIENT_ID',
        'KEYCLOAK_SERVER_URL',
        'KEYCLOAK_REALM_NAME'
    ]

    return all(os.getenv(var) for var in required_vars)
```

#### 4.4.2 UI Integration

The frontend will poll the license status endpoint and display warnings using the existing banner component pattern:

```typescript
// New component: LicenseWarningBanner.tsx
interface LicenseStatus {
  warn: boolean;
  message: string;
}

export function LicenseWarningBanner() {
  const [licenseStatus, setLicenseStatus] = useState<LicenseStatus>({ warn: false, message: "" });

  useEffect(() => {
    const checkLicenseStatus = async () => {
      try {
        const response = await fetch('/api/license-status');
        const status = await response.json();
        setLicenseStatus(status);
      } catch (error) {
        console.error('Failed to check license status:', error);
      }
    };

    // Check immediately and then every hour
    checkLicenseStatus();
    const interval = setInterval(checkLicenseStatus, 60 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  if (!licenseStatus.warn) {
    return null;
  }

  return (
    <div className="bg-red-600 text-white p-4 rounded flex items-center justify-between">
      <div className="flex items-center">
        <FaExclamationTriangle className="mr-3" />
        <span>{licenseStatus.message}</span>
      </div>
    </div>
  );
}
```

### 4.5 Cronjob Configuration

The cronjob configurations will be deployed via the OpenHands-Cloud helm charts.

#### 4.5.1 Collection Cronjob

The collection cronjob runs weekly to gather metrics:

```yaml
# charts/openhands/templates/telemetry-collection-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: {{ include "openhands.fullname" . }}-telemetry-collection
  labels:
    {{- include "openhands.labels" . | nindent 4 }}
spec:
  schedule: "0 2 * * 0"  # Weekly on Sunday at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: telemetry-collector
            image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
            env:
            {{- include "openhands.env" . | nindent 12 }}
            command:
            - python
            - -c
            - |
              from enterprise.storage.maintenance_task import MaintenanceTask, MaintenanceTaskStatus
              from enterprise.storage.database import session_maker
              from enterprise.server.telemetry.collection_processor import TelemetryCollectionProcessor

              # Create collection task
              processor = TelemetryCollectionProcessor()
              task = MaintenanceTask()
              task.set_processor(processor)
              task.status = MaintenanceTaskStatus.PENDING

              with session_maker() as session:
                  session.add(task)
                  session.commit()
          restartPolicy: OnFailure
```

#### 4.5.2 Upload Cronjob

The upload cronjob runs daily to send metrics to Replicated:

```yaml
# charts/openhands/templates/telemetry-upload-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: {{ include "openhands.fullname" . }}-telemetry-upload
  labels:
    {{- include "openhands.labels" . | nindent 4 }}
spec:
  schedule: "0 3 * * *"  # Daily at 3 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: telemetry-uploader
            image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
            env:
            {{- include "openhands.env" . | nindent 12 }}
            - name: REPLICATED_PUBLISHABLE_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "openhands.fullname" . }}-replicated-config
                  key: publishable-key
            - name: REPLICATED_APP_SLUG
              value: {{ .Values.telemetry.replicatedAppSlug | default "openhands-enterprise" | quote }}
            command:
            - python
            - -c
            - |
              from enterprise.storage.maintenance_task import MaintenanceTask, MaintenanceTaskStatus
              from enterprise.storage.database import session_maker
              from enterprise.server.telemetry.upload_processor import TelemetryUploadProcessor
              import os

              # Create upload task
              processor = TelemetryUploadProcessor(
                  replicated_publishable_key=os.getenv('REPLICATED_PUBLISHABLE_KEY'),
                  replicated_app_slug=os.getenv('REPLICATED_APP_SLUG', 'openhands-enterprise')
              )
              task = MaintenanceTask()
              task.set_processor(processor)
              task.status = MaintenanceTaskStatus.PENDING

              with session_maker() as session:
                  session.add(task)
                  session.commit()
          restartPolicy: OnFailure
```

## 5. Implementation Plan

All implementation must pass existing lints and tests. New functionality requires comprehensive unit tests with >90% coverage. Integration tests should verify end-to-end telemetry flow including collection, storage, upload, and warning display.

### 5.1 Database Schema and Models (M1)

**Repository**: OpenHands
Establish the foundational database schema and SQLAlchemy models for telemetry data storage.

#### 5.1.1 OpenHands - Database Migration

- [ ] `enterprise/migrations/versions/077_create_telemetry_tables.py`
- [ ] `enterprise/storage/telemetry_metrics.py`
- [ ] `enterprise/storage/telemetry_config.py`

#### 5.1.2 OpenHands - Model Tests

- [ ] `enterprise/tests/unit/storage/test_telemetry_metrics.py`
- [ ] `enterprise/tests/unit/storage/test_telemetry_config.py`

**Demo**: Database tables created and models can store/retrieve telemetry data.

### 5.2 Metrics Collection Framework (M2)

**Repository**: OpenHands
Implement the pluggable metrics collection system with registry and base classes.

#### 5.2.1 OpenHands - Core Collection Framework

- [ ] `enterprise/server/telemetry/__init__.py`
- [ ] `enterprise/server/telemetry/collector_base.py`
- [ ] `enterprise/server/telemetry/collector_registry.py`
- [ ] `enterprise/server/telemetry/decorators.py`

#### 5.2.2 OpenHands - Example Collectors

- [ ] `enterprise/server/telemetry/collectors/__init__.py`
- [ ] `enterprise/server/telemetry/collectors/system_metrics.py`
- [ ] `enterprise/server/telemetry/collectors/user_activity.py`

#### 5.2.3 OpenHands - Framework Tests

- [ ] `enterprise/tests/unit/telemetry/test_collector_base.py`
- [ ] `enterprise/tests/unit/telemetry/test_collector_registry.py`
- [ ] `enterprise/tests/unit/telemetry/test_system_metrics.py`

**Demo**: Developers can create new collectors with a single file change using the @register_collector decorator.

### 5.3 Collection and Upload Processors (M3)

**Repository**: OpenHands
Implement maintenance task processors for collecting metrics and uploading to Replicated.

#### 5.3.1 OpenHands - Collection Processor

- [ ] `enterprise/server/telemetry/collection_processor.py`
- [ ] `enterprise/tests/unit/telemetry/test_collection_processor.py`

#### 5.3.2 OpenHands - Upload Processor

- [ ] `enterprise/server/telemetry/upload_processor.py`
- [ ] `enterprise/tests/unit/telemetry/test_upload_processor.py`

#### 5.3.3 OpenHands - Integration Tests

- [ ] `enterprise/tests/integration/test_telemetry_flow.py`

**Demo**: Metrics are automatically collected weekly and uploaded daily to Replicated vendor portal.

### 5.4 License Warning API (M4)

**Repository**: OpenHands
Implement the license status endpoint for the warning system.

#### 5.4.1 OpenHands - License Status API

- [ ] `enterprise/server/routes/license.py`
- [ ] `enterprise/tests/unit/routes/test_license.py`

#### 5.4.2 OpenHands - API Integration

- [ ] Update `enterprise/saas_server.py` to include license router

**Demo**: License status API returns warning status based on telemetry upload success.

### 5.5 UI Warning Banner (M5)

**Repository**: OpenHands
Implement the frontend warning banner component and integration.

#### 5.5.1 OpenHands - UI Warning Banner

- [ ] `frontend/src/components/features/license/license-warning-banner.tsx`
- [ ] `frontend/src/components/features/license/license-warning-banner.test.tsx`

#### 5.5.2 OpenHands - UI Integration

- [ ] Update main UI layout to include license warning banner
- [ ] Add license status polling service

**Demo**: License warnings appear in UI when telemetry uploads fail for >4 days, with accurate expiration countdown.

### 5.6 Helm Chart Deployment Configuration (M6)

**Repository**: OpenHands-Cloud
Create Kubernetes cronjob configurations and deployment scripts.

#### 5.6.1 OpenHands-Cloud - Cronjob Manifests

- [ ] `charts/openhands/templates/telemetry-collection-cronjob.yaml`
- [ ] `charts/openhands/templates/telemetry-upload-cronjob.yaml`

#### 5.6.2 OpenHands-Cloud - Configuration Management

- [ ] `charts/openhands/templates/replicated-secret.yaml`
- [ ] Update `charts/openhands/values.yaml` with telemetry configuration options:
  ```yaml
  # Add to values.yaml
  telemetry:
    enabled: true
    replicatedAppSlug: "openhands-enterprise"
    adminEmail: ""  # Optional: admin email for customer identification

  # Add to deployment environment variables
  env:
    OPENHANDS_ADMIN_EMAIL: "{{ .Values.telemetry.adminEmail }}"
  ```

**Demo**: Complete telemetry system deployed via helm chart with configurable collection intervals and Replicated integration.

### 5.7 Documentation and Enhanced Collectors (M7)

**Repository**: OpenHands
Add comprehensive metrics collectors, monitoring capabilities, and documentation.

#### 5.7.1 OpenHands - Advanced Collectors

- [ ] `enterprise/server/telemetry/collectors/conversation_metrics.py`
- [ ] `enterprise/server/telemetry/collectors/integration_usage.py`
- [ ] `enterprise/server/telemetry/collectors/performance_metrics.py`

#### 5.7.2 OpenHands - Monitoring and Testing

- [ ] `enterprise/server/telemetry/monitoring.py`
- [ ] `enterprise/tests/e2e/test_telemetry_system.py`
- [ ] Performance tests for large-scale metric collection

#### 5.7.3 OpenHands - Technical Documentation

- [ ] `enterprise/server/telemetry/README.md`
- [ ] Update deployment documentation with telemetry configuration instructions
- [ ] Add troubleshooting guide for telemetry issues

**Demo**: Rich telemetry data flowing to vendor portal with comprehensive monitoring, alerting for system health, and complete documentation.
