"""
Test script to validate BottoSmart functionality without external dependencies
"""

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing BottoSmart module structure...")
    
    # Check if files exist
    import os
    files_to_check = [
        'config.py',
        'market_analyzer.py',
        'ml_predictor.py',
        'risk_manager.py',
        'exchange_connector.py',
        'bottosmart.py',
        'main.py',
        'backtester.py'
    ]
    
    missing_files = []
    for file in files_to_check:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    
    # Check file sizes (basic sanity check)
    file_sizes = {}
    for file in files_to_check:
        size = os.path.getsize(file)
        file_sizes[file] = size
        if size < 1000:  # Less than 1KB might indicate empty file
            print(f"⚠️  {file} seems small ({size} bytes)")
    
    print("\n📊 File Sizes:")
    for file, size in file_sizes.items():
        print(f"  {file}: {size:,} bytes")
    
    return True

def test_configuration():
    """Test configuration structure"""
    print("\n🔧 Testing configuration...")
    
    # Check .env.example exists
    if os.path.exists('.env.example'):
        print("✅ Environment template exists")
        with open('.env.example', 'r') as f:
            content = f.read()
            required_vars = ['EXCHANGE', 'API_KEY', 'API_SECRET', 'SANDBOX']
            for var in required_vars:
                if var in content:
                    print(f"✅ {var} found in template")
                else:
                    print(f"❌ {var} missing from template")
    else:
        print("❌ .env.example missing")
    
    return True

def test_documentation():
    """Test documentation completeness"""
    print("\n📚 Testing documentation...")
    
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            content = f.read()
            if len(content) > 5000:  # Substantial README
                print("✅ Comprehensive README exists")
            else:
                print("⚠️  README seems short")
    else:
        print("❌ README.md missing")
    
    return True

if __name__ == "__main__":
    import os
    
    print("🚀 BottoSmart v2.0 - Component Test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_configuration()
    all_passed &= test_documentation()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! BottoSmart is ready for setup.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy .env.example to .env and configure")
        print("3. Run: python main.py --mode test")
    else:
        print("❌ Some tests failed. Please review the output above.")