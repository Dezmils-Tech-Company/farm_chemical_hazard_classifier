"""
src/disease_mapper.py
Step 4: Map crop diseases to chemical treatments (Bridge Model 1 → Model 2)
"""

import pandas as pd
from typing import List, Dict, Tuple

class DiseaseChemicalMapper:
    """Map crop diseases to recommended pesticides"""
    
    def __init__(self):
        self.disease_map = self._create_disease_map()
        
    def _create_disease_map(self) -> Dict:
        """Create mapping table for Kenyan crops"""
        
        return {
            # MAIZE DISEASES (most common in Kenya)
            'maize_leaf_blight': {
                'common_name': 'Maize Leaf Blight',
                'pathogen': 'Fungal (Exserohilum turcicum)',
                'chemicals': ['mancozeb', 'azoxystrobin', 'propiconazole'],
                'organic_alternative': 'Neem oil spray + remove infected leaves',
                'min_confidence': 0.70
            },
            'maize_rust': {
                'common_name': 'Maize Rust',
                'pathogen': 'Fungal (Puccinia sorghi)',
                'chemicals': ['tebuconazole', 'propiconazole', 'azoxystrobin'],
                'organic_alternative': 'Sulfur spray + compost tea',
                'min_confidence': 0.70
            },
            'maize_streak_virus': {
                'common_name': 'Maize Streak Virus',
                'pathogen': 'Viral (transmitted by leafhoppers)',
                'chemicals': ['imidacloprid', 'acetamiprid'],
                'organic_alternative': 'Remove infected plants + neem oil for vectors',
                'min_confidence': 0.80
            },
            'maize_smut': {
                'common_name': 'Maize Smut',
                'pathogen': 'Fungal (Ustilago maydis)',
                'chemicals': ['mancozeb', 'chlorothalonil'],
                'organic_alternative': 'Remove galls + crop rotation',
                'min_confidence': 0.75
            },
            
            # TOMATO DISEASES
            'tomato_late_blight': {
                'common_name': 'Tomato Late Blight',
                'pathogen': 'Fungal (Phytophthora infestans)',
                'chemicals': ['mancozeb', 'chlorothalonil', 'metalaxyl'],
                'organic_alternative': 'Copper spray + remove infected leaves',
                'min_confidence': 0.75
            },
            'tomato_early_blight': {
                'common_name': 'Tomato Early Blight',
                'pathogen': 'Fungal (Alternaria solani)',
                'chemicals': ['chlorothalonil', 'azoxystrobin', 'mancozeb'],
                'organic_alternative': 'Neem oil + baking soda spray',
                'min_confidence': 0.70
            },
            'tomato_leaf_curl': {
                'common_name': 'Tomato Leaf Curl',
                'pathogen': 'Viral (whitefly transmitted)',
                'chemicals': ['imidacloprid', 'deltamethrin'],
                'organic_alternative': 'Yellow sticky traps + neem oil',
                'min_confidence': 0.80
            },
            'tomato_bacterial_wilt': {
                'common_name': 'Tomato Bacterial Wilt',
                'pathogen': 'Bacterial (Ralstonia solanacearum)',
                'chemicals': ['copper_hydroxide', 'streptomycin'],
                'organic_alternative': 'Crop rotation + solarization',
                'min_confidence': 0.75
            },
            
            # POTATO DISEASES
            'potato_late_blight': {
                'common_name': 'Potato Late Blight',
                'pathogen': 'Fungal (Phytophthora infestans)',
                'chemicals': ['mancozeb', 'metalaxyl', 'chlorothalonil'],
                'organic_alternative': 'Copper hydroxide + hilling soil',
                'min_confidence': 0.75
            },
            'potato_early_blight': {
                'common_name': 'Potato Early Blight',
                'pathogen': 'Fungal (Alternaria solani)',
                'chemicals': ['azoxystrobin', 'chlorothalonil', 'mancozeb'],
                'organic_alternative': 'Neem oil + crop rotation',
                'min_confidence': 0.70
            },
            
            # BEAN DISEASES
            'bean_rust': {
                'common_name': 'Bean Rust',
                'pathogen': 'Fungal (Uromyces appendiculatus)',
                'chemicals': ['tebuconazole', 'azoxystrobin', 'mancozeb'],
                'organic_alternative': 'Sulfur + compost tea',
                'min_confidence': 0.70
            },
            'bean_anthracnose': {
                'common_name': 'Bean Anthracnose',
                'pathogen': 'Fungal (Colletotrichum lindemuthianum)',
                'chemicals': ['chlorothalonil', 'mancozeb', 'azoxystrobin'],
                'organic_alternative': 'Copper spray + resistant varieties',
                'min_confidence': 0.75
            },
            'bean_angular_leaf_spot': {
                'common_name': 'Bean Angular Leaf Spot',
                'pathogen': 'Bacterial (Pseudomonas syringae)',
                'chemicals': ['copper_hydroxide'],
                'organic_alternative': 'Remove infected leaves + copper spray',
                'min_confidence': 0.70
            },
            
            # CASSAVA DISEASES
            'cassava_mosaic': {
                'common_name': 'Cassava Mosaic',
                'pathogen': 'Viral (whitefly transmitted)',
                'chemicals': ['imidacloprid', 'acetamiprid'],
                'organic_alternative': 'Use certified virus-free cuttings',
                'min_confidence': 0.85
            },
            'cassava_brown_streak': {
                'common_name': 'Cassava Brown Streak',
                'pathogen': 'Viral',
                'chemicals': [],
                'organic_alternative': 'Rogue infected plants + resistant varieties',
                'min_confidence': 0.80
            },
            
            # VEGETABLE DISEASES
            'cabbage_black_rot': {
                'common_name': 'Cabbage Black Rot',
                'pathogen': 'Bacterial (Xanthomonas campestris)',
                'chemicals': ['copper_hydroxide'],
                'organic_alternative': 'Crop rotation + resistant varieties',
                'min_confidence': 0.75
            },
            'onion_downy_mildew': {
                'common_name': 'Onion Downy Mildew',
                'pathogen': 'Fungal (Peronospora destructor)',
                'chemicals': ['mancozeb', 'metalaxyl', 'chlorothalonil'],
                'organic_alternative': 'Good air circulation + neem oil',
                'min_confidence': 0.70
            },
            'pepper_anthracnose': {
                'common_name': 'Pepper Anthracnose',
                'pathogen': 'Fungal (Colletotrichum species)',
                'chemicals': ['chlorothalonil', 'azoxystrobin', 'mancozeb'],
                'organic_alternative': 'Copper spray + remove infected fruits',
                'min_confidence': 0.75
            }
        }
    
    def get_chemicals(self, disease_name: str) -> Tuple[List[str], str]:
        """Get recommended chemicals for a disease"""
        
        disease_key = disease_name.lower().replace(' ', '_')
        
        if disease_key not in self.disease_map:
            return [], f"Disease '{disease_name}' not found in database"
        
        info = self.disease_map[disease_key]
        return info['chemicals'], info['organic_alternative']
    
    def get_all_diseases(self) -> List[str]:
        """Get list of all supported diseases"""
        return [info['common_name'] for info in self.disease_map.values()]
    
    def get_disease_info(self, disease_name: str) -> Dict:
        """Get detailed information about a disease"""
        disease_key = disease_name.lower().replace(' ', '_')
        return self.disease_map.get(disease_key, None)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert mapping to DataFrame for analysis"""
        rows = []
        for key, info in self.disease_map.items():
            for chem in info['chemicals']:
                rows.append({
                    'disease_key': key,
                    'disease_name': info['common_name'],
                    'pathogen_type': info['pathogen'],
                    'chemical': chem,
                    'organic_alternative': info['organic_alternative']
                })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    mapper = DiseaseChemicalMapper()
    
    print("="*60)
    print("DISEASE-CHEMICAL MAPPING")
    print("="*60)
    
    # Test with a disease
    disease = "maize leaf blight"
    chemicals, organic = mapper.get_chemicals(disease)
    
    print(f"\n🌽 Disease: {disease}")
    print(f"   Recommended chemicals: {chemicals}")
    print(f"   Organic alternative: {organic}")
    
    print(f"\n📊 Total diseases: {len(mapper.disease_map)}")
    print(f"   Total chemical mappings: {len(mapper.to_dataframe())}")