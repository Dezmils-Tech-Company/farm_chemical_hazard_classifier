"""
src/complete_system.py
Step 5: Complete system integrating Model 1 + Model 2
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from safety_predictor import SafetyPredictor
from disease_mapper import DiseaseChemicalMapper

class CompleteSafetySystem:
    """
    Complete Thesis System: Disease Detection → Safety Screening
    
    Model 1 (CNN): Detects crop disease from image
    Model 2 (Your Model): Predicts chemical toxicity
    """
    
    def __init__(self):
        print("="*70)
        print("COMPLETE AGRI-CHEM SAFETY SYSTEM")
        print("Model 1 (CNN Disease Detection) + Model 2 (Toxicity Classifier)")
        print("="*70)
        
        # Initialize components
        self.predictor = SafetyPredictor()
        self.mapper = DiseaseChemicalMapper()
        
        print("\n✅ System ready for use")
    
    def process_disease(self, disease_name: str, confidence: float = 0.85):
        """
        Complete pipeline: Disease → Chemical recommendations → Safety check
        
        Args:
            disease_name: e.g., "maize leaf blight"
            confidence: Model 1 confidence score (simulated)
        """
        
        print(f"\n{'='*60}")
        print("🌾 CROP DISEASE ANALYSIS")
        print(f"{'='*60}")
        
        # Step 1: Display disease detection (Model 1 output)
        print(f"\n📸 MODEL 1 - CNN DISEASE DETECTION")
        print(f"   Disease: {disease_name.upper()}")
        print(f"   Confidence: {confidence*100:.1f}%")
        
        # Step 2: Get chemical treatments
        print(f"\n🔍 Looking up treatments...")
        chemicals, organic = self.mapper.get_chemicals(disease_name)
        
        if not chemicals:
            print(f"   ⚠️ No chemical treatments found")
            print(f"\n🌱 RECOMMENDATION:")
            print(f"   {organic}")
            return {
                'disease': disease_name,
                'chemicals': [],
                'organic': organic,
                'recommendations': []
            }
        
        print(f"   Found {len(chemicals)} potential treatments")
        
        # Step 3: Check safety of each chemical (Model 2)
        print(f"\n⚕️ MODEL 2 - TOXICITY SCREENING")
        print(f"   {'Chemical':<20} {'WHO Class':<10} {'Safety':<15} {'Action':<10}")
        print(f"   {'-'*55}")
        
        recommendations = []
        for chem in chemicals:
            safety = self.predictor.predict(chem)
            
            if not safety['error']:
                recommendations.append(safety)
                
                # Display results
                status_icon = {
                    'APPROVED': '✅',
                    'CAUTION': '⚠️',
                    'WARNING': '⚠️',
                    'BLOCKED': '🔴'
                }.get(safety['action'], '❓')
                
                print(f"   {chem:<20} {safety['who_class']:<10} {safety['safety_level']:<15} {status_icon} {safety['action']}")
        
        # Step 4: Generate final recommendations
        print(f"\n{'='*60}")
        print("📋 FINAL RECOMMENDATIONS")
        print(f"{'='*60}")
        
        safe_recs = [r for r in recommendations if r['action'] in ['APPROVED', 'CAUTION']]
        warning_recs = [r for r in recommendations if r['action'] == 'WARNING']
        blocked_recs = [r for r in recommendations if r['action'] == 'BLOCKED']
        
        # Display safe options
        if safe_recs:
            print(f"\n✅ SAFE OPTIONS (Recommended):")
            for rec in safe_recs:
                print(f"   • {rec['chemical']} - {rec['safety_level']}")
                print(f"     PPE: {rec['ppe_required']}")
                print(f"     Re-entry: {rec['reentry_hours']} hours")
        
        # Display caution options
        if warning_recs:
            print(f"\n⚠️ USE WITH CAUTION (Only if safe options unavailable):")
            for rec in warning_recs:
                print(f"   • {rec['chemical']} - {rec['safety_level']}")
                print(f"     PPE: {rec['ppe_required']}")
        
        # Display blocked options
        if blocked_recs:
            print(f"\n🔴 DO NOT USE (Blocked by safety system):")
            for rec in blocked_recs:
                print(f"   • {rec['chemical']} - {rec['safety_level']}")
        
        # Organic alternative
        print(f"\n🌱 ORGANIC ALTERNATIVE:")
        print(f"   {organic}")
        
        return {
            'disease': disease_name,
            'confidence': confidence,
            'chemicals_checked': len(recommendations),
            'safe_options': len(safe_recs),
            'warning_options': len(warning_recs),
            'blocked_options': len(blocked_recs),
            'organic_alternative': organic,
            'recommendations': recommendations
        }
    
    def compare_treatments(self, disease_name: str):
        """Compare all treatments for a disease"""
        chemicals, _ = self.mapper.get_chemicals(disease_name)
        
        if not chemicals:
            print(f"No treatments found for {disease_name}")
            return
        
        print(f"\n{'='*60}")
        print(f"TREATMENT COMPARISON: {disease_name.upper()}")
        print(f"{'='*60}")
        
        comparison = []
        for chem in chemicals:
            safety = self.predictor.predict(chem)
            if not safety['error']:
                comparison.append({
                    'Chemical': chem,
                    'WHO Class': safety['who_class'],
                    'Safety Level': safety['safety_level'],
                    'Action': safety['action'],
                    'PPE': safety['ppe_required'][:30] + '...',
                    'Confidence': f"{safety['confidence']:.0f}%"
                })
        
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        
        return df
    
    def interactive_session(self):
        """Run an interactive session for testing"""
        print("\n" + "="*60)
        print("INTERACTIVE SAFETY SYSTEM")
        print("="*60)
        print("\nAvailable diseases:")
        diseases = list(self.mapper.disease_map.keys())
        for i, disease in enumerate(diseases[:10], 1):
            print(f"   {i}. {disease.replace('_', ' ').title()}")
        print(f"   ... and {len(diseases)-10} more")
        
        while True:
            print("\n" + "-"*40)
            disease = input("\nEnter disease name (or 'quit'): ").strip().lower()
            
            if disease == 'quit':
                print("Goodbye!")
                break
            
            result = self.process_disease(disease)
            
            if result['chemicals_checked'] == 0:
                print("\n⚠️ No treatments found for this disease. Try: maize leaf blight, tomato late blight, bean rust")


if __name__ == "__main__":
    system = CompleteSafetySystem()
    
    # Test with example diseases
    print("\n" + "="*70)
    print("TESTING THE COMPLETE SYSTEM")
    print("="*70)
    
    # Test disease 1: Maize Leaf Blight
    system.process_disease("maize leaf blight", confidence=0.87)
    
    # Test disease 2: Tomato Late Blight
    print("\n" + "="*70)
    system.process_disease("tomato late blight", confidence=0.92)
    
    # Compare treatments
    system.compare_treatments("maize leaf blight")
    
    # Uncomment for interactive mode
    # system.interactive_session()