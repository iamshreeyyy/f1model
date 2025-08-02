#!/bin/bash

# F1 AI System - API Test Script
echo "🧪 Testing F1 AI Prediction System APIs"
echo "========================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test API endpoint
test_endpoint() {
    local endpoint=$1
    local description=$2
    
    echo -n "Testing $description... "
    
    if response=$(curl -s -f "http://localhost:8000$endpoint"); then
        echo -e "${GREEN}✓ PASS${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        return 1
    fi
}

# Test endpoints
echo "Testing Backend API Endpoints:"
echo "------------------------------"

test_endpoint "/" "Root endpoint"
test_endpoint "/api/health" "Health check"
test_endpoint "/api/model-stats" "Model statistics"
test_endpoint "/api/predictions" "Race predictions"
test_endpoint "/api/race-results" "Race results"
test_endpoint "/api/driver-stats" "Driver statistics"
test_endpoint "/api/championships" "Championships"
test_endpoint "/api/analytics" "Analytics data"

echo ""
echo "Testing Custom Prediction:"
echo "-------------------------"

# Test custom prediction
echo -n "Testing custom race prediction... "
if response=$(curl -s -f -X POST "http://localhost:8000/api/predict-race" \
  -H "Content-Type: application/json" \
  -d '{
    "drivers": ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Lando Norris"],
    "circuit": "Monaco",
    "weather": {"AirTemp": 25, "TrackTemp": 35, "Humidity": 60, "WindSpeed": 5, "Rainfall": 0}
  }'); then
    echo -e "${GREEN}✓ PASS${NC}"
    echo "Sample prediction result:"
    echo "$response" | python3 -m json.tool | head -20
else
    echo -e "${RED}✗ FAIL${NC}"
fi

echo ""
echo "Testing Frontend Connectivity:"
echo "-----------------------------"

# Test frontend
for port in 3000 3001 3002; do
    echo -n "Testing frontend on port $port... "
    if curl -s -f "http://localhost:$port" > /dev/null; then
        echo -e "${GREEN}✓ PASS${NC} - Frontend accessible at http://localhost:$port"
        frontend_found=true
        break
    else
        echo -e "${YELLOW}- Not on port $port${NC}"
    fi
done

if [ -z "$frontend_found" ]; then
    echo -e "${RED}✗ Frontend not accessible on any common port${NC}"
fi

echo ""
echo "System URLs:"
echo "------------"
echo "🌐 Frontend Dashboard: http://localhost:3001"
echo "⚡ Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/api/health"

echo ""
echo "Quick API Examples:"
echo "------------------"
echo "curl http://localhost:8000/api/model-stats"
echo "curl http://localhost:8000/api/predictions"
echo ""
echo "curl -X POST http://localhost:8000/api/predict-race \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"drivers\": [\"Max Verstappen\", \"Lewis Hamilton\"], \"circuit\": \"Monaco\"}'"
