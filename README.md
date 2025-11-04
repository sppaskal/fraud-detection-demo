# Fraud Document Detection Demo

This is a demo Python script that uses Tesseract OCR to detect suspicious areas in PDF and image files.
Detection is based on several heuristics:
1. Existence of template markers ('template', 'sample', 'void', ect)
2. Low confidence from OCR
3. Large font size deviations

The script processes all files inside of src/files and outputs png files for each inside of src/files/results
with overlays that show any points of suspicion. Additionally, the script outputs a suspicion score to the
console from 0 to 1.

Overlay colors:
- Red: template markers
- Blue: low confidence
- Orange: font size deviation

Follow these steps to run the script locally using Docker.

## Prerequisites
- Docker installed (Docker Desktop for Windows/macOS, or Docker and Docker Compose for Linux).
- Git installed to clone the repository.

## Setup Instructions
1. **Install Docker**:
   - For Windows/macOS: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
   - For Linux: Install Docker and Docker Compose (e.g., `sudo apt install docker.io docker-compose` on Ubuntu).

2. **Start Docker**:
   - On Windows/macOS: Launch Docker Desktop and ensure it’s running (verify with `docker --version` in a terminal).
   - On Linux: Start the Docker daemon: `sudo systemctl start docker`.

3. **Clone the Repository**:
   ```bash
   git clone https://github.com/sppaskal/fraud-detection-demo
   ```

4. **Navigate to the Project Directory**:
   ```bash
   cd fraud-detection-demo
   ```

5. **Build the image**:
   ```bash
   docker-compose build
   ```

6. **Run the script**:
   ```bash
   docker-compose up
   ```