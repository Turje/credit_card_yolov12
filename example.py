"""
Example script showing how to use the Roboflow downloader.
"""
from src import RoboflowDownloader, config

# Example: Download a credit card dataset
# Replace with your actual workspace, project, and version
if __name__ == "__main__":
    # Check if API key is set
    if not config.validate_api_key():
        print("Error: ROBOFLOW_API_KEY not set!")
        print("Set it with: export ROBOFLOW_API_KEY=your_key")
        exit(1)
    
    # Initialize downloader
    downloader = RoboflowDownloader()
    
    # Example download (uncomment and fill in your details)
    # downloader.download_dataset(
    #     workspace="your-workspace",
    #     project="credit-card",
    #     version=1,
    #     format="yolov8",
    #     location="datasets"
    # )
    
    print("Example script ready!")
    print("Uncomment the download_dataset call and fill in your details to use.")

