"""
test_api_endpoints.py
Complete test script for Agricultural Chemical Safety AI API
Run this after starting the API server
"""

import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000/api"
API_URL = "http://localhost:8000"

# Colors for console output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}📌 {msg}{Colors.RESET}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️ {msg}{Colors.RESET}")

def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")

def print_subheader(title: str):
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'─'*40}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{title:^40}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'─'*40}{Colors.RESET}")

class APITester:
    def __init__(self):
        self.base_url = BASE_URL
        self.api_url = API_URL
        self.test_results = []
        self.start_time = None
        self.end_time = None
    
    def record_result(self, test_name: str, passed: bool, message: str = ""):
        """Record test result"""
        self.test_results.append({
            "name": test_name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        if passed:
            print_success(f"{test_name}: {message if message else 'Passed'}")
        else:
            print_error(f"{test_name}: {message if message else 'Failed'}")
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        print_header("TEST 1: HEALTH CHECK")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print(f"   Model loaded: {data.get('model_loaded')}")
                print(f"   Database size: {data.get('database_size')}")
                print(f"   API status: {data.get('status')}")
                print(f"   Version: {data.get('version')}")
                self.record_result("Health Check", True, "API is healthy")
                return True
            else:
                print_error(f"Status: {response.status_code}")
                self.record_result("Health Check", False, f"HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print_error(f"Cannot connect to API at {self.base_url}")
            print_info("Make sure the API is running: python api/main.py")
            self.record_result("Health Check", False, "Connection error")
            return False
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("Health Check", False, str(e))
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint"""
        print_header("TEST 2: ROOT ENDPOINT")
        
        try:
            response = requests.get(self.api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print(f"   Message: {data.get('message')}")
                print(f"   Version: {data.get('version')}")
                endpoints = data.get('endpoints', {})
                print(f"   Endpoints: {len(endpoints)}")
                for name, url in list(endpoints.items())[:5]:
                    print(f"      - {name}: {url}")
                self.record_result("Root Endpoint", True, "API root accessible")
                return True
            else:
                print_error(f"Status: {response.status_code}")
                self.record_result("Root Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("Root Endpoint", False, str(e))
            return False
    
    def test_single_prediction(self) -> bool:
        """Test single chemical prediction"""
        print_header("TEST 3: SINGLE CHEMICAL PREDICTION")
        
        test_cases = [
            {"chemical_name": "glyphosate", "expected_class": "U", "expected_action": "APPROVED"},
            {"chemical_name": "chlorpyrifos", "expected_class": "II", "expected_action": "WARNING"},
            {"chemical_name": "malathion", "expected_class": "III", "expected_action": "CAUTION"},
            {"chemical_name": "carbofuran", "expected_class": "Ib", "expected_action": "BLOCKED"},
            {"chemical_name": "mancozeb", "expected_class": "U", "expected_action": "APPROVED"},
            {"chemical_name": "atrazine", "expected_class": "III", "expected_action": "CAUTION"},
            {"chemical_name": "cypermethrin", "expected_class": "II", "expected_action": "WARNING"},
        ]
        
        all_passed = True
        passed_count = 0
        
        for test in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json={"chemical_name": test['chemical_name']},
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('success'):
                        predicted = data.get('who_class')
                        action = data.get('action')
                        confidence = data.get('confidence', 0)
                        
                        if predicted == test['expected_class']:
                            print_success(f"   {test['chemical_name']}: {predicted} ({confidence:.1f}%) → {action} ✓")
                            passed_count += 1
                        else:
                            print_warning(f"   {test['chemical_name']}: Predicted {predicted}, Expected {test['expected_class']}")
                            all_passed = False
                        
                        # Print additional info
                        print(f"      Safety: {data.get('safety_level')}")
                        print(f"      Re-entry: {data.get('reentry_hours')} hours")
                        print(f"      PPE: {data.get('ppe_required', '')[:60]}...")
                    else:
                        print_error(f"   {test['chemical_name']}: {data.get('error_message')}")
                        all_passed = False
                else:
                    print_error(f"   {test['chemical_name']}: HTTP {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.Timeout:
                print_error(f"   {test['chemical_name']}: Timeout")
                all_passed = False
            except Exception as e:
                print_error(f"   {test['chemical_name']}: {e}")
                all_passed = False
        
        success_rate = passed_count / len(test_cases) * 100
        self.record_result("Single Prediction", all_passed, f"{passed_count}/{len(test_cases)} correct ({success_rate:.0f}%)")
        return all_passed
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        print_header("TEST 4: BATCH PREDICTION")
        
        chemicals = ["glyphosate", "chlorpyrifos", "mancozeb", "atrazine", "carbofuran"]
        
        try:
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json={"chemicals": chemicals},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                total = data.get('total', 0)
                results = data.get('results', [])
                
                print_success(f"Batch processed: {total} chemicals")
                print(f"\n   Results:")
                for result in results:
                    if result.get('success'):
                        print(f"      ✅ {result['chemical']:15} → {result.get('who_class')} ({result.get('safety_level')})")
                    else:
                        print(f"      ❌ {result.get('chemical')}: {result.get('error_message')}")
                
                self.record_result("Batch Prediction", True, f"Processed {total} chemicals")
                return True
            else:
                print_error(f"HTTP {response.status_code}")
                self.record_result("Batch Prediction", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("Batch Prediction", False, str(e))
            return False
    
    def test_disease_recommendation(self) -> bool:
        """Test disease recommendation endpoint"""
        print_header("TEST 5: DISEASE RECOMMENDATION")
        
        test_diseases = [
            "maize leaf blight",
            "tomato late blight",
            "bean rust",
            "potato late blight",
            "cassava mosaic"
        ]
        
        all_passed = True
        
        for disease in test_diseases:
            print_subheader(f"Disease: {disease}")
            
            try:
                response = requests.post(
                    f"{self.base_url}/disease/recommend",
                    json={"disease_name": disease, "confidence": 0.85},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('success'):
                        safe_count = len(data.get('safe_recommendations', []))
                        caution_count = len(data.get('caution_recommendations', []))
                        blocked_count = len(data.get('blocked_recommendations', []))
                        
                        print_success(f"Found treatments for {disease}:")
                        print(f"   ✅ Safe: {safe_count}")
                        print(f"   ⚠️ Caution: {caution_count}")
                        print(f"   🔴 Blocked: {blocked_count}")
                        print(f"   🌱 Organic: {data.get('organic_alternative', 'N/A')[:60]}...")
                        
                        # Show safe recommendations
                        if safe_count > 0:
                            print(f"\n   Recommended chemicals:")
                            for rec in data.get('safe_recommendations', [])[:3]:
                                print(f"      → {rec['chemical']} ({rec['who_class']}): {rec['confidence']:.0f}% confidence")
                                print(f"        PPE: {rec.get('ppe', 'N/A')[:50]}...")
                    else:
                        print_error(f"   {data.get('error_message')}")
                        all_passed = False
                else:
                    print_error(f"   HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print_error(f"   Error: {e}")
                all_passed = False
        
        self.record_result("Disease Recommendation", all_passed, f"Tested {len(test_diseases)} diseases")
        return all_passed
    
    def test_list_diseases(self) -> bool:
        """Test list diseases endpoint"""
        print_header("TEST 6: LIST DISEASES")
        
        try:
            # Test without filters
            response = requests.get(f"{self.base_url}/diseases", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                total = data.get('total', 0)
                diseases = data.get('diseases', [])
                
                print_success(f"Total diseases: {total}")
                print(f"\n   First 10 diseases:")
                for disease in diseases[:10]:
                    print(f"      • {disease.get('name')} ({disease.get('chemical_count')} chemicals)")
                
                # Test with crop filter
                print(f"\n   Testing crop filter (maize):")
                response2 = requests.get(f"{self.base_url}/diseases?crop_type=maize", timeout=10)
                if response2.status_code == 200:
                    data2 = response2.json()
                    print(f"      Maize diseases: {data2.get('total')}")
                
                # Test with search
                print(f"\n   Testing search (blight):")
                response3 = requests.get(f"{self.base_url}/diseases?search=blight", timeout=10)
                if response3.status_code == 200:
                    data3 = response3.json()
                    print(f"      Found: {data3.get('total')} diseases")
                    for d in data3.get('diseases', [])[:3]:
                        print(f"      → {d.get('name')}")
                
                self.record_result("List Diseases", True, f"Found {total} diseases")
                return True
            else:
                print_error(f"HTTP {response.status_code}")
                self.record_result("List Diseases", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("List Diseases", False, str(e))
            return False
    
    def test_crop_types(self) -> bool:
        """Test crop types endpoint"""
        print_header("TEST 7: CROP TYPES")
        
        try:
            response = requests.get(f"{self.base_url}/diseases/crops", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print_success("Crop types retrieved")
                print(f"   Total diseases: {data.get('total_diseases')}")
                print(f"\n   Disease counts by crop:")
                crops = data.get('crops', {})
                for crop, count in crops.items():
                    if count > 0:
                        print(f"      • {crop.title()}: {count} diseases")
                
                self.record_result("Crop Types", True, f"Found {len([c for c in crops.values() if c > 0])} crop types")
                return True
            else:
                print_error(f"HTTP {response.status_code}")
                self.record_result("Crop Types", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("Crop Types", False, str(e))
            return False
    
    def test_disease_detail(self) -> bool:
        """Test disease detail endpoint"""
        print_header("TEST 8: DISEASE DETAIL")
        
        test_disease = "maize_leaf_blight"
        
        try:
            response = requests.get(f"{self.base_url}/diseases/{test_disease}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print_success(f"Disease: {data.get('common_name')}")
                    print(f"   Pathogen: {data.get('pathogen')}")
                    print(f"   Chemicals: {', '.join(data.get('chemicals', []))}")
                    print(f"   Organic: {data.get('organic_alternative', 'N/A')[:60]}...")
                    
                    # Show chemical safety
                    chem_safety = data.get('chemical_safety', [])
                    if chem_safety:
                        print(f"\n   Chemical Safety:")
                        for cs in chem_safety[:3]:
                            print(f"      → {cs['chemical']}: {cs['who_class']} ({cs['action']}) - {cs['confidence']:.0f}%")
                    
                    self.record_result("Disease Detail", True, f"Retrieved {data.get('common_name')}")
                    return True
                else:
                    print_error(data.get('error_message'))
                    self.record_result("Disease Detail", False, data.get('error_message'))
                    return False
            else:
                print_error(f"HTTP {response.status_code}")
                self.record_result("Disease Detail", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("Disease Detail", False, str(e))
            return False
    
    def test_disease_chemicals(self) -> bool:
        """Test disease chemicals endpoint"""
        print_header("TEST 9: DISEASE CHEMICALS")
        
        test_disease = "tomato_late_blight"
        
        try:
            response = requests.get(f"{self.base_url}/diseases/{test_disease}/chemicals", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print_success(f"Disease: {data.get('disease')}")
                    print(f"   Pathogen: {data.get('pathogen')}")
                    print(f"   Total chemicals: {data.get('total_chemicals')}")
                    print(f"   Safe: {data.get('safe_count')}, Warning: {data.get('warning_count')}, Blocked: {data.get('blocked_count')}")
                    print(f"   Chemicals: {', '.join(data.get('chemicals', []))}")
                    
                    self.record_result("Disease Chemicals", True, f"Found {data.get('total_chemicals')} chemicals")
                    return True
                else:
                    print_error(data.get('error_message'))
                    self.record_result("Disease Chemicals", False, data.get('error_message'))
                    return False
            else:
                print_error(f"HTTP {response.status_code}")
                self.record_result("Disease Chemicals", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("Disease Chemicals", False, str(e))
            return False
    
    def test_list_chemicals(self) -> bool:
        """Test list chemicals endpoint"""
        print_header("TEST 10: LIST CHEMICALS")
        
        try:
            # Test with limit
            response = requests.get(f"{self.base_url}/chemicals?limit=20", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                total = data.get('total', 0)
                chemicals = data.get('chemicals', [])
                
                print_success(f"Total chemicals in database: {total}")
                print(f"   Showing first 20: {', '.join(chemicals[:20])}")
                
                # Test with search
                print(f"\n   Testing search (chlor):")
                response2 = requests.get(f"{self.base_url}/chemicals?search=chlor&limit=10", timeout=10)
                if response2.status_code == 200:
                    data2 = response2.json()
                    found = data2.get('chemicals', [])
                    print(f"      Found: {', '.join(found)}")
                
                self.record_result("List Chemicals", True, f"Found {total} chemicals")
                return True
            else:
                print_error(f"HTTP {response.status_code}")
                self.record_result("List Chemicals", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("List Chemicals", False, str(e))
            return False
    
    def test_invalid_chemical(self) -> bool:
        """Test handling of invalid chemical name"""
        print_header("TEST 11: INVALID CHEMICAL HANDLING")
        
        invalid_chemical = "nonexistent_chemical_xyz123"
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"chemical_name": invalid_chemical},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if not data.get('success') and data.get('error_message'):
                    print_success(f"Properly handled invalid chemical")
                    print(f"   Error message: {data.get('error_message')}")
                    self.record_result("Invalid Chemical", True, "Gracefully handled")
                    return True
                else:
                    print_warning(f"Returned success for invalid chemical")
                    self.record_result("Invalid Chemical", False, "Should return error")
                    return False
            else:
                print_error(f"HTTP {response.status_code}")
                self.record_result("Invalid Chemical", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("Invalid Chemical", False, str(e))
            return False
    
    def test_invalid_disease(self) -> bool:
        """Test handling of invalid disease name"""
        print_header("TEST 12: INVALID DISEASE HANDLING")
        
        invalid_disease = "nonexistent_disease_xyz"
        
        try:
            response = requests.post(
                f"{self.base_url}/disease/recommend",
                json={"disease_name": invalid_disease},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if not data.get('success') and data.get('error_message'):
                    print_success(f"Properly handled invalid disease")
                    print(f"   Error message: {data.get('error_message')[:100]}...")
                    self.record_result("Invalid Disease", True, "Gracefully handled")
                    return True
                else:
                    print_warning(f"Returned success for invalid disease")
                    self.record_result("Invalid Disease", False, "Should return error")
                    return False
            else:
                print_error(f"HTTP {response.status_code}")
                self.record_result("Invalid Disease", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"Error: {e}")
            self.record_result("Invalid Disease", False, str(e))
            return False
    
    def test_response_time(self) -> bool:
        """Test API response times"""
        print_header("TEST 13: RESPONSE TIME PERFORMANCE")
        
        endpoints = [
            ("GET", "/api/health", None),
            ("POST", "/api/predict", {"chemical_name": "glyphosate"}),
            ("GET", "/api/diseases", None),
            ("POST", "/api/disease/recommend", {"disease_name": "maize leaf blight"}),
        ]
        
        all_good = True
        results = []
        
        for method, path, data in endpoints:
            url = f"{self.api_url}{path}"
            
            start_time = time.time()
            
            try:
                if method == "GET":
                    response = requests.get(url, timeout=15)
                else:
                    response = requests.post(url, json=data, timeout=15)
                
                elapsed = (time.time() - start_time) * 1000
                
                status = "✅" if elapsed < 1000 else "⚠️" if elapsed < 2000 else "❌"
                
                if elapsed < 1000:
                    print_success(f"{method} {path}: {elapsed:.0f}ms")
                elif elapsed < 2000:
                    print_warning(f"{method} {path}: {elapsed:.0f}ms")
                    all_good = False
                else:
                    print_error(f"{method} {path}: {elapsed:.0f}ms")
                    all_good = False
                
                results.append({"endpoint": path, "time_ms": elapsed, "status": status})
                
            except Exception as e:
                print_error(f"{method} {path}: Error - {e}")
                all_good = False
        
        # Print summary
        avg_time = sum(r["time_ms"] for r in results) / len(results) if results else 0
        print(f"\n   Average response time: {avg_time:.0f}ms")
        
        self.record_result("Response Time", all_good, f"Avg: {avg_time:.0f}ms")
        return all_good
    
    def run_all_tests(self):
        """Run all tests and print summary"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}SAFE_SHAMBA AI - API TEST SUITE{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"\nTesting API at: {Colors.BLUE}{self.base_url}{Colors.RESET}")
        print(f"Make sure the API is running: {Colors.YELLOW}python api/main.py{Colors.RESET}")
        
        # Wait for user confirmation
        input(f"\n{Colors.YELLOW}Press Enter to start testing...{Colors.RESET}")
        
        self.start_time = time.time()
        
        # Run all tests
        tests = [
            ("Health Check", self.test_health_check),
            ("Root Endpoint", self.test_root_endpoint),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Disease Recommendation", self.test_disease_recommendation),
            ("List Diseases", self.test_list_diseases),
            ("Crop Types", self.test_crop_types),
            ("Disease Detail", self.test_disease_detail),
            ("Disease Chemicals", self.test_disease_chemicals),
            ("List Chemicals", self.test_list_chemicals),
            ("Invalid Chemical Handling", self.test_invalid_chemical),
            ("Invalid Disease Handling", self.test_invalid_disease),
            ("Response Time Performance", self.test_response_time),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print_error(f"Test '{test_name}' crashed: {e}")
                self.record_result(test_name, False, f"Crashed: {e}")
        
        self.end_time = time.time()
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print_header("TEST SUMMARY")
        
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        duration = self.end_time - self.start_time
        
        for result in self.test_results:
            if result["passed"]:
                print_success(f"{result['name']}")
            else:
                print_error(f"{result['name']} - {result['message']}")
        
        print(f"\n{Colors.BOLD}{'─'*40}{Colors.RESET}")
        print(f"{Colors.BOLD}RESULTS: {passed}/{total} tests passed{Colors.RESET}")
        print(f"{Colors.BOLD}Duration: {duration:.1f} seconds{Colors.RESET}")
        
        if passed == total:
            print_success(f"\n🎉 All tests passed! Your API is working perfectly!")
        else:
            print_warning(f"\n⚠️ {total - passed} test(s) failed. Check the errors above.")
        
        print(f"\n📚 API Documentation: {Colors.BLUE}http://localhost:8000/docs{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")


def quick_test():
    """Quick minimal test for API"""
    print(f"\n{Colors.BOLD}Quick API Test{Colors.RESET}")
    
    try:
        # Health check
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print_success("API is running")
        else:
            print_error(f"API returned {r.status_code}")
            return False
        
        # Test prediction
        r = requests.post(f"{BASE_URL}/predict", json={"chemical_name": "glyphosate"}, timeout=10)
        if r.status_code == 200 and r.json().get('success'):
            data = r.json()
            print_success(f"Prediction: {data['chemical']} → {data['who_class']} ({data['confidence']:.0f}%)")
        else:
            print_error("Prediction failed")
            return False
        
        # Test disease
        r = requests.post(f"{BASE_URL}/disease/recommend", json={"disease_name": "maize leaf blight"}, timeout=10)
        if r.status_code == 200 and r.json().get('success'):
            data = r.json()
            print_success(f"Disease: {len(data['safe_recommendations'])} safe options found")
        else:
            print_error("Disease recommendation failed")
            return False
        
        print_success("\n✅ API is working correctly!")
        return True
        
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Make sure it's running: python api/main.py")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Check if quick test requested
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        tester = APITester()
        tester.run_all_tests()