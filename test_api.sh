#!/bin/bash

# API Testing Script
# Tests all major backend endpoints

BASE_URL="http://localhost:8000"

echo "🧪 Testing No-Code ML Training Platform API"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test counter
PASSED=0
FAILED=0

# Test function
test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    
    echo -n "Testing $name... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        echo -e "${GREEN}✓ PASSED${NC} (HTTP $http_code)"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $http_code)"
        echo "  Response: $body"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Start tests
echo "1. Health Check Tests"
echo "---------------------"
test_endpoint "Health endpoint" "GET" "/"
echo ""

echo "2. Model Recommendation Tests"
echo "-----------------------------"
test_endpoint "Get model candidates" "GET" "/model-candidates?task=classification"
test_endpoint "Get search spaces" "GET" "/api/hyperopt/search-spaces"
echo ""

echo "3. Training Management Tests"
echo "---------------------------"
test_endpoint "List training runs" "GET" "/api/training/list"
echo ""

echo "4. Hyperparameter Tests"
echo "----------------------"
test_endpoint "Suggest search space" "POST" "/api/hyperopt/suggest-space" \
    '{"dataset_size": 1000, "task_type": "classification", "model_size": "base"}'
echo ""

# Summary
echo "==========================================="
echo "Test Summary:"
echo -e "  ${GREEN}Passed: $PASSED${NC}"
echo -e "  ${RED}Failed: $FAILED${NC}"
echo "==========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi