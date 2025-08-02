#!/bin/bash

# F1 AI System - Complete Setup and Deployment Script
# This script sets up both backend and frontend

echo "🏎️  F1 AI Prediction System - Setup & Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_header "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    print_status "Python 3: $(python3 --version)"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed."
        exit 1
    fi
    print_status "Node.js: $(node --version)"
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is required but not installed."
        exit 1
    fi
    print_status "npm: $(npm --version)"
    
    print_status "All requirements satisfied!"
}

# Setup backend
setup_backend() {
    print_header "Setting up F1 AI Backend..."
    
    cd f1-ai-backend
    
    # Create virtual environment
    print_status "Creating Python virtual environment..."
    python3 -m venv f1_env
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source f1_env/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Train initial model
    print_status "Training initial ML model..."
    python model.py
    
    print_status "Backend setup complete!"
    cd ..
}

# Setup frontend
setup_frontend() {
    print_header "Setting up F1 Dashboard Frontend..."
    
    cd f1-dashboard
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    # Create environment file if it doesn't exist
    if [ ! -f .env.local ]; then
        print_status "Creating environment configuration..."
        echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
    fi
    
    print_status "Frontend setup complete!"
    cd ..
}

# Start services
start_services() {
    print_header "Starting F1 AI Services..."
    
    # Start backend
    print_status "Starting backend API server..."
    cd f1-ai-backend
    source f1_env/bin/activate
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..
    
    # Wait for backend to start
    print_status "Waiting for backend to initialize..."
    sleep 5
    
    # Test backend
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_status "Backend API is running on http://localhost:8000"
    else
        print_error "Backend failed to start properly"
        return 1
    fi
    
    # Start frontend
    print_status "Starting frontend development server..."
    cd f1-dashboard
    nohup npm run dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to start
    print_status "Waiting for frontend to initialize..."
    sleep 10
    
    # Save PIDs
    echo $BACKEND_PID > backend.pid
    echo $FRONTEND_PID > frontend.pid
    
    print_status "Services started successfully!"
}

# Stop services
stop_services() {
    print_header "Stopping F1 AI Services..."
    
    if [ -f backend.pid ]; then
        BACKEND_PID=$(cat backend.pid)
        if ps -p $BACKEND_PID > /dev/null; then
            kill $BACKEND_PID
            print_status "Backend stopped"
        fi
        rm -f backend.pid
    fi
    
    if [ -f frontend.pid ]; then
        FRONTEND_PID=$(cat frontend.pid)
        if ps -p $FRONTEND_PID > /dev/null; then
            kill $FRONTEND_PID
            print_status "Frontend stopped"
        fi
        rm -f frontend.pid
    fi
    
    # Kill any remaining processes
    pkill -f "uvicorn main:app"
    pkill -f "next dev"
    
    print_status "All services stopped"
}

# Show service status
show_status() {
    print_header "F1 AI System Status"
    
    # Check backend
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_status "✅ Backend API: Running (http://localhost:8000)"
        print_status "   📊 API Documentation: http://localhost:8000/docs"
    else
        print_warning "❌ Backend API: Not running"
    fi
    
    # Check frontend (try common ports)
    for port in 3000 3001 3002; do
        if curl -f http://localhost:$port > /dev/null 2>&1; then
            print_status "✅ Frontend Dashboard: Running (http://localhost:$port)"
            break
        fi
    done
    
    if ! curl -f http://localhost:3000 > /dev/null 2>&1 && \
       ! curl -f http://localhost:3001 > /dev/null 2>&1 && \
       ! curl -f http://localhost:3002 > /dev/null 2>&1; then
        print_warning "❌ Frontend Dashboard: Not running"
    fi
    
    echo ""
    print_status "System Information:"
    print_status "   🐍 Python: $(python3 --version)"
    print_status "   🟢 Node.js: $(node --version)"
    print_status "   📦 npm: $(npm --version)"
    
    if [ -f f1-ai-backend/data/models/f1_model.pkl ]; then
        print_status "   🤖 ML Model: Trained and ready"
    else
        print_warning "   🤖 ML Model: Not found - run 'python model.py' in backend directory"
    fi
}

# Show help
show_help() {
    echo "F1 AI Prediction System - Deployment Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Complete system setup (backend + frontend)"
    echo "  start     - Start both services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Show system status"
    echo "  backend   - Setup backend only"
    echo "  frontend  - Setup frontend only"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Complete setup"
    echo "  $0 start     # Start services"
    echo "  $0 status    # Check status"
}

# Handle different commands
case "${1:-setup}" in
    "setup")
        check_requirements
        setup_backend
        setup_frontend
        start_services
        echo ""
        print_status "🎉 F1 AI System setup complete!"
        print_status "🌐 Frontend: http://localhost:3001 (or 3000)"
        print_status "⚡ Backend API: http://localhost:8000"
        print_status "📚 API Docs: http://localhost:8000/docs"
        ;;
    "start")
        start_services
        show_status
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_services
        show_status
        ;;
    "status")
        show_status
        ;;
    "backend")
        check_requirements
        setup_backend
        ;;
    "frontend")
        check_requirements
        setup_frontend
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
