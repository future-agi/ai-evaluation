#!/usr/bin/env python
"""
Setup script for SDK integration testing against the backend.

This script creates the necessary test data (organization, user, API key)
in the backend database for running SDK integration tests.

Usage:
    # From core-backend directory with test env loaded:
    cd /path/to/core-backend
    set -a && source .env.test.local && set +a
    python /path/to/ai-evaluation/python/scripts/setup_integration_test.py

    # Or copy this script to core-backend and run:
    python setup_integration_test.py
"""

import os
import sys
import django


def setup_django():
    """Setup Django environment."""
    # Add core-backend to path if not already there
    backend_path = os.environ.get('CORE_BACKEND_PATH')
    if backend_path and backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()


def create_eval_templates():
    """Create evaluation templates for testing."""
    from model_hub.models.evals_metric import EvalTemplate

    templates_to_create = [
        {
            "name": "groundedness",
            "description": "Evaluate if the output is grounded in the provided context",
            "eval_id": 1001,
            "eval_tags": ["rag", "groundedness"],
            "config": {
                "eval_type_id": "Groundedness",
                "required_keys": ["context", "response"],
                "optional_keys": [],
            },
            "owner": "system",
            "organization": None,
        },
        {
            "name": "toxicity",
            "description": "Detect toxic content in text",
            "eval_id": 1002,
            "eval_tags": ["safety", "toxicity"],
            "config": {
                "eval_type_id": "Toxicity",
                "required_keys": ["text"],
                "optional_keys": [],
            },
            "owner": "system",
            "organization": None,
        },
        {
            "name": "pii",
            "description": "Detect personally identifiable information",
            "eval_id": 1003,
            "eval_tags": ["safety", "pii"],
            "config": {
                "eval_type_id": "PII",
                "required_keys": ["text"],
                "optional_keys": [],
            },
            "owner": "system",
            "organization": None,
        },
        {
            "name": "is_polite",
            "description": "Evaluate if text is polite",
            "eval_id": 1004,
            "eval_tags": ["tone", "politeness"],
            "config": {
                "eval_type_id": "IsPolite",
                "required_keys": ["input"],
                "optional_keys": [],
            },
            "owner": "system",
            "organization": None,
        },
        {
            "name": "is_helpful",
            "description": "Evaluate if response is helpful",
            "eval_id": 1005,
            "eval_tags": ["quality", "helpfulness"],
            "config": {
                "eval_type_id": "IsHelpful",
                "required_keys": ["input", "output"],
                "optional_keys": [],
            },
            "owner": "system",
            "organization": None,
        },
    ]

    print("\n5. Evaluation Templates:")
    created_count = 0
    for template_data in templates_to_create:
        template, created = EvalTemplate.objects.get_or_create(
            name=template_data["name"],
            defaults=template_data,
        )
        status = "Created" if created else "Exists"
        print(f"   {status}: {template.name} (eval_id: {template.eval_id})")
        if created:
            created_count += 1

    print(f"   Total: {EvalTemplate.objects.count()} templates in database")
    return created_count


def create_test_data():
    """Create test organization, user, and API key."""
    from accounts.models import Organization, User, OrgApiKey
    from accounts.models.workspace import Workspace
    from tfc.constants.roles import OrganizationRoles

    # Test credentials - these will be used by SDK integration tests
    TEST_EMAIL = "sdk_test@futureagi.com"
    TEST_PASSWORD = "sdk_test_password_123"
    TEST_API_KEY = "test_api_key_12345"
    TEST_SECRET_KEY = "test_secret_key_67890"

    print("=" * 60)
    print("SDK Integration Test Setup")
    print("=" * 60)

    # 1. Create or get organization
    org, org_created = Organization.objects.get_or_create(
        name="SDK Test Organization"
    )
    print(f"\n1. Organization: {org.name}")
    print(f"   Created: {org_created}")
    print(f"   ID: {org.id}")

    # 2. Create or get user
    user = User.objects.filter(email=TEST_EMAIL).first()
    if not user:
        user = User.objects.create_user(
            email=TEST_EMAIL,
            password=TEST_PASSWORD,
            name="SDK Test User",
            organization=org,
            organization_role=OrganizationRoles.OWNER,
        )
        user_created = True
    else:
        user_created = False

    print(f"\n2. User: {user.email}")
    print(f"   Created: {user_created}")
    print(f"   ID: {user.id}")

    # 3. Create or get workspace
    workspace, ws_created = Workspace.objects.get_or_create(
        organization=org,
        is_default=True,
        defaults={
            "name": "SDK Test Workspace",
            "is_active": True,
            "created_by": user,
        }
    )
    print(f"\n3. Workspace: {workspace.name}")
    print(f"   Created: {ws_created}")
    print(f"   ID: {workspace.id}")

    # 4. Create or get API key
    api_key, key_created = OrgApiKey.objects.get_or_create(
        api_key=TEST_API_KEY,
        defaults={
            "organization": org,
            "user": user,
            "secret_key": TEST_SECRET_KEY,
            "name": "SDK Integration Test Key",
            "enabled": True,
            "type": "user",
        }
    )

    # Update if exists but has different values
    if not key_created:
        api_key.secret_key = TEST_SECRET_KEY
        api_key.enabled = True
        api_key.save()

    print(f"\n4. API Key:")
    print(f"   Created: {key_created}")
    print(f"   API Key: {api_key.api_key}")
    print(f"   Secret Key: {api_key.secret_key}")
    print(f"   Enabled: {api_key.enabled}")

    # Print environment variables for SDK tests
    print("\n" + "=" * 60)
    print("SDK Integration Test Environment Variables")
    print("=" * 60)
    print(f"""
# Add these to your shell or .env file:
export FI_API_KEY="{TEST_API_KEY}"
export FI_SECRET_KEY="{TEST_SECRET_KEY}"
export FI_BASE_URL="http://localhost:8001"

# Or use in Python:
from fi.evals import Evaluator

evaluator = Evaluator(
    fi_api_key="{TEST_API_KEY}",
    fi_secret_key="{TEST_SECRET_KEY}",
    fi_base_url="http://localhost:8001"
)
""")

    # Print test user credentials for JWT auth testing
    print("=" * 60)
    print("Test User Credentials (for JWT auth)")
    print("=" * 60)
    print(f"""
Email: {TEST_EMAIL}
Password: {TEST_PASSWORD}

# Get JWT token:
curl -X POST http://localhost:8001/accounts/token/ \\
  -H "Content-Type: application/json" \\
  -d '{{"email": "{TEST_EMAIL}", "password": "{TEST_PASSWORD}"}}'
""")

    return {
        "organization": org,
        "user": user,
        "workspace": workspace,
        "api_key": api_key,
        "credentials": {
            "api_key": TEST_API_KEY,
            "secret_key": TEST_SECRET_KEY,
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
        }
    }


def verify_setup():
    """Verify the setup by testing API key authentication."""
    from accounts.models import OrgApiKey

    print("\n" + "=" * 60)
    print("Verifying Setup")
    print("=" * 60)

    try:
        key = OrgApiKey.objects.get(api_key="test_api_key_12345")
        print(f"✓ API Key found: {key.api_key}")
        print(f"✓ Organization: {key.organization.name}")
        print(f"✓ User: {key.user.email}")
        print(f"✓ Enabled: {key.enabled}")
        return True
    except OrgApiKey.DoesNotExist:
        print("✗ API Key not found!")
        return False


if __name__ == "__main__":
    setup_django()
    create_test_data()
    create_eval_templates()
    verify_setup()

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("""
Next steps:
1. Start backend server (if not running):
   python manage.py runserver 0.0.0.0:8001

2. Run SDK integration tests:
   cd /path/to/ai-evaluation/python
   export FI_API_KEY="test_api_key_12345"
   export FI_SECRET_KEY="test_secret_key_67890"
   export FI_BASE_URL="http://localhost:8001"
   pytest tests/integration/ -v
""")
