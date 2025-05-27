import os

def create_app_directories():
    """Create the necessary directory structure for the Flask app"""
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created templates directory")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Created static directory")
    
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        print("Created uploads directory")
    
    print("Directory structure is now set up correctly.")
    print("Please move index.html to the templates folder.")

if __name__ == "__main__":
    create_app_directories()