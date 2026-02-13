"""
Cleanup SageMaker endpoints to save costs.

This script deletes SageMaker endpoints, endpoint configurations,
and optionally models.
"""

import argparse
import boto3
from botocore.exceptions import ClientError


def delete_endpoint(endpoint_name: str, delete_config: bool = True, delete_model: bool = False):
    """
    Delete SageMaker endpoint and optionally its configuration and model.
    
    Args:
        endpoint_name: Name of the endpoint to delete
        delete_config: Whether to delete endpoint configuration
        delete_model: Whether to delete the model
    """
    client = boto3.client('sagemaker')
    
    print(f"Deleting endpoint: {endpoint_name}")
    
    try:
        # Get endpoint details before deletion
        endpoint = client.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint['EndpointConfigName']
        
        # Delete endpoint
        print(f"  Deleting endpoint...")
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f"  ✓ Endpoint deleted")
        
        if delete_config:
            # Get config details
            try:
                config = client.describe_endpoint_config(EndpointConfigName=config_name)
                model_name = config['ProductionVariants'][0]['ModelName']
                
                # Delete endpoint configuration
                print(f"  Deleting endpoint configuration: {config_name}")
                client.delete_endpoint_config(EndpointConfigName=config_name)
                print(f"  ✓ Endpoint configuration deleted")
                
                if delete_model:
                    # Delete model
                    print(f"  Deleting model: {model_name}")
                    client.delete_model(ModelName=model_name)
                    print(f"  ✓ Model deleted")
                    
            except ClientError as e:
                print(f"  Warning: Could not delete config/model: {str(e)}")
        
        print(f"✓ Cleanup complete for {endpoint_name}\n")
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"  Endpoint '{endpoint_name}' not found\n")
        else:
            print(f"  Error: {str(e)}\n")


def list_endpoints(prefix: str = "nids"):
    """List all SageMaker endpoints with given prefix."""
    client = boto3.client('sagemaker')
    
    print(f"Listing endpoints with prefix '{prefix}':")
    
    try:
        response = client.list_endpoints(
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=100
        )
        
        endpoints = [
            ep for ep in response['Endpoints']
            if ep['EndpointName'].startswith(prefix)
        ]
        
        if not endpoints:
            print(f"  No endpoints found with prefix '{prefix}'\n")
            return []
        
        for ep in endpoints:
            status = ep['EndpointStatus']
            name = ep['EndpointName']
            created = ep['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"  - {name}")
            print(f"    Status: {status}")
            print(f"    Created: {created}")
        
        print()
        return [ep['EndpointName'] for ep in endpoints]
        
    except ClientError as e:
        print(f"  Error listing endpoints: {str(e)}\n")
        return []


def main():
    parser = argparse.ArgumentParser(description="Cleanup SageMaker endpoints")
    
    parser.add_argument(
        "--endpoint-name",
        type=str,
        help="Specific endpoint name to delete"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all NIDS endpoints"
    )
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete all NIDS endpoints (prefix: nids)"
    )
    parser.add_argument(
        "--delete-config",
        action="store_true",
        default=True,
        help="Delete endpoint configuration (default: True)"
    )
    parser.add_argument(
        "--delete-model",
        action="store_true",
        help="Delete model (default: False)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAGEMAKER ENDPOINT CLEANUP")
    print("=" * 60 + "\n")
    
    if args.list:
        list_endpoints("nids")
        return
    
    if args.delete_all:
        endpoints = list_endpoints("nids")
        
        if not endpoints:
            return
        
        if not args.yes:
            response = input(f"Delete {len(endpoints)} endpoint(s)? (yes/no): ")
            if response.lower() != "yes":
                print("Cancelled.")
                return
        
        for endpoint_name in endpoints:
            delete_endpoint(
                endpoint_name,
                delete_config=args.delete_config,
                delete_model=args.delete_model
            )
    
    elif args.endpoint_name:
        if not args.yes:
            response = input(f"Delete endpoint '{args.endpoint_name}'? (yes/no): ")
            if response.lower() != "yes":
                print("Cancelled.")
                return
        
        delete_endpoint(
            args.endpoint_name,
            delete_config=args.delete_config,
            delete_model=args.delete_model
        )
    
    else:
        print("Please specify --endpoint-name, --delete-all, or --list")
        print("Run with --help for usage information")


if __name__ == "__main__":
    main()
