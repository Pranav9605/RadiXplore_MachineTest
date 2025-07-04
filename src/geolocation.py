
import os
# Limit all threading to 1 to prevent segfaults and semaphore leaks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple, List
from pandas import DataFrame
from src.utils import call_gemini
import re

class GeoLocator:
    def __init__(self, gazetteer: DataFrame, gemini_key: str, threshold: float = 0.5):  # Lowered threshold
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.gaz = gazetteer
        self.key = gemini_key
        self.threshold = threshold

   
        required_cols = ["place_name", "latitude", "longitude"]
        missing_cols = [col for col in required_cols if col not in gazetteer.columns]
        if missing_cols:
            raise ValueError(f"Gazetteer missing columns: {missing_cols}")

        names = gazetteer["place_name"].tolist()
        embs = self.embedder.encode(names, convert_to_numpy=True)
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs.astype(np.float32))
        
        print(f" GeoLocator initialized with {len(names)} places, threshold={threshold}")

    def infer(self, context: str, project_name: str = None) -> Optional[Tuple[float, float]]:
        """
        Infer coordinates from context, with enhanced debugging and fallback strategies
        """
        print(f"\n Inferring location for context: '{context[:100]}...'")

        if project_name:
            direct_match = self._direct_gazetteer_search(project_name)
            if direct_match:
                print(f" Direct match found for '{project_name}': {direct_match}")
                return direct_match
        
        # Strategy 2: Semantic similarity search
        similarity_match = self._semantic_search(context)
        if similarity_match:
            print(f" Semantic match found: {similarity_match}")
            return similarity_match
        
        # Strategy 3: Extract location names from context
        location_names = self._extract_location_names(context)
        if location_names:
            for loc_name in location_names:
                direct_match = self._direct_gazetteer_search(loc_name)
                if direct_match:
                    print(f" Extracted location match for '{loc_name}': {direct_match}")
                    return direct_match
        

        print(" Falling back to Gemini API...")
        gemini_result = self._fallback_gemini(context, project_name)
        if gemini_result:
            print(f" Gemini result: {gemini_result}")
            return gemini_result
        
        print(" No coordinates found")
        return None

    def _direct_gazetteer_search(self, name: str) -> Optional[Tuple[float, float]]:
        """Search for exact or partial matches in gazetteer"""
        if not name:
            return None
            

        exact_match = self.gaz[self.gaz["place_name"].str.lower() == name.lower()]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return float(row.latitude), float(row.longitude)
 
        partial_match = self.gaz[self.gaz["place_name"].str.contains(name, case=False, na=False)]
        if not partial_match.empty:
            row = partial_match.iloc[0]
            return float(row.latitude), float(row.longitude)
        
        return None

    def _semantic_search(self, context: str) -> Optional[Tuple[float, float]]:
        """Use FAISS for semantic similarity search"""
        try:
            q = self.embedder.encode([context])
            faiss.normalize_L2(q)
            scores, idxs = self.index.search(q.astype(np.float32), 3)  # Get top 3 matches
            
            print(f" Top similarity scores: {scores[0]}")
            
            if scores[0][0] >= self.threshold:
                row = self.gaz.iloc[idxs[0][0]]
                print(f" Best match: '{row.place_name}' (score: {scores[0][0]:.3f})")
                return float(row.latitude), float(row.longitude)
            else:
                print(f" Best similarity score {scores[0][0]:.3f} below threshold {self.threshold}")
                
        except Exception as e:
            print(f" Semantic search error: {e}")
        
        return None

    def _extract_location_names(self, context: str) -> List[str]:
        """Extract potential location names from context"""
        # Look for common location patterns
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:mine|project|deposit|field)\b',
            r'\b(?:located|at|in|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+)\s+(?:WA|Western Australia|Australia)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:region|area|district)\b'
        ]
        
        locations = []
        for pattern in patterns:
            matches = re.findall(pattern, context)
            locations.extend(matches)
        
        # Remove duplicates and common non-location words
        stop_words = {'Figure', 'Unit', 'Court', 'Park', 'Ltd', 'Resources', 'Mining', 'Gold', 'Star'}
        locations = list(set([loc for loc in locations if loc not in stop_words]))
        
        if locations:
            print(f" Extracted potential locations: {locations}")
        
        return locations

    def _fallback_gemini(self, context: str, project_name: str = None) -> Optional[Tuple[float, float]]:
        """Enhanced Gemini fallback with better prompt"""
        enhanced_context = f"Project: {project_name}\nContext: {context}" if project_name else context
        
        prompt = f"""Extract the most specific latitude and longitude coordinates from this Australian mining context:

{enhanced_context}

Focus on:
- Mining project locations in Western Australia
- Specific place names, towns, or geographic features
- Consider that many mines are in remote areas of WA

Respond ONLY in valid JSON format:
{{"latitude": <decimal_degrees>, "longitude": <decimal_degrees>}}

If you cannot determine coordinates, respond with:
{{"latitude": null, "longitude": null}}"""

        try:
            txt = call_gemini(prompt, self.key)
            print(f" Gemini response: {txt}")
            

            txt = txt.strip()
            if txt.startswith('```'):
                txt = txt.split('\n')[1:-1]  # Remove first and last lines
                txt = '\n'.join(txt)
            
            coords = json.loads(txt)
            lat, lon = coords.get("latitude"), coords.get("longitude")
            
            if lat is not None and lon is not None:
                # Validate coordinates are reasonable for Australia
                if -45 <= lat <= -10 and 110 <= lon <= 155:
                    return float(lat), float(lon)
                else:
                    print(f" Invalid coordinates for Australia: {lat}, {lon}")
            
        except json.JSONDecodeError as e:
            print(f" JSON decode error: {e}")
            print(f"Raw response: {txt}")
        except Exception as e:
            print(f" Gemini API error: {e}")
        
        return None