import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import requests_cache

session = requests_cache.CachedSession('wikidata_cache', backend='filesystem', expire_after=None)

@dataclass
class WikidataMatch:
    id: str
    label: str
    description: str
    url: str
    birthplace: Optional[str] = None
    occupation: Optional[str] = None
    birthyear: Optional[str] = None

class EntityMatchError(Exception):
    pass

def get_entity_label(entity_id: str) -> str:
    """Get label for an entity ID"""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    response = session.get(url)
    data = response.json()['entities'][entity_id]
    return data.get('labels', {}).get('en', {}).get('value', entity_id)

def extract_entity_details(entity_data: dict, entity_id: str) -> dict:
    """Extract birthplace, occupation, and birth year from entity data"""
    claims = entity_data.get('claims', {})
    details = {}
    
    # Extract birth year from P569 (date of birth)
    if 'P569' in claims:
        try:
            date = claims['P569'][0]['mainsnak']['datavalue']['value']['time']
            details['birthyear'] = date.split('-')[0].replace('+', '')
        except (KeyError, IndexError):
            details['birthyear'] = None
            
    # Extract birthplace from P19 (place of birth)
    if 'P19' in claims:
        try:
            birthplace_id = claims['P19'][0]['mainsnak']['datavalue']['value']['id']
            details['birthplace'] = get_entity_label(birthplace_id)
        except (KeyError, IndexError):
            details['birthplace'] = None
            
    # Extract occupation from P106 (occupation)
    if 'P106' in claims:
        try:
            occupation_id = claims['P106'][0]['mainsnak']['datavalue']['value']['id']
            details['occupation'] = get_entity_label(occupation_id)
        except (KeyError, IndexError):
            details['occupation'] = None
            
    return details

def match_entity_strict(entity: str) -> WikidataMatch:
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity,
        "limit": 5,
        "type": "item"
    }
    
    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        matches = response.json().get("search", [])
        
        if not matches:
            raise EntityMatchError(f"No matches found for '{entity}'")
        
        exact_matches = []
        for m in matches:
            if m.get("label", "").lower() == entity.lower():
                entity_data = session.get(
                    f"https://www.wikidata.org/wiki/Special:EntityData/{m['id']}.json"
                ).json()['entities'][m['id']]
                
                # Check if entity is instance of human (Q5)
                claims = entity_data.get('claims', {})
                instance_of = claims.get('P31', [])
                
                if any(claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id') == 'Q5' 
                       for claim in instance_of):
                    exact_matches.append((m, entity_data))
        
        if len(exact_matches) == 0:
            raise EntityMatchError(f"No exact person matches found for '{entity}'")
        if len(exact_matches) > 1:
            raise EntityMatchError(f"Multiple exact person matches found for '{entity}'")
            
        match, entity_data = exact_matches[0]
        details = extract_entity_details(entity_data, match['id'])
        
        return WikidataMatch(
            id=match["id"],
            label=match["label"],
            description=match.get("description", ""),
            url=f"https://www.wikidata.org/wiki/{match['id']}",
            birthplace=details.get('birthplace'),
            occupation=details.get('occupation'),
            birthyear=details.get('birthyear')
        )
        
    except requests.RequestException as e:
        raise EntityMatchError(f"API error: {str(e)}")

def process_entity_file(filename: str) -> tuple[Dict[str, WikidataMatch], Dict[str, str]]:
    """
    Process entities and return both matches and errors.
    Returns:
        Tuple of (successful matches dict, errors dict)
    """
    matches = {}
    errors = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        entities = [line.strip() for line in f if line.strip()]
    
    for entity in tqdm(entities):
        try:
            matches[entity] = match_entity_strict(entity)
        except EntityMatchError as e:
            errors[entity] = str(e)
            
    return matches, errors

def analyze_wikidata_entities(matches: Dict[str, WikidataMatch]):
    """Analyze common attributes across matched Wikidata entities"""
    attributes = {
        'id': 0, 'label': 0, 'description': 0, 'url': 0,
        'birthplace': 0, 'occupation': 0, 'birthyear': 0
    }
    total_entities = len(matches)
    
    for entity, match in matches.items():
        for attr in attributes:
            if hasattr(match, attr) and getattr(match, attr):
                attributes[attr] += 1
                
    # Convert to percentages
    for attr in attributes:
        attributes[attr] = (attributes[attr] / total_entities) * 100
        
    return attributes

if __name__ == "__main__":
    matches, errors = process_entity_file("src/prompt_entities.txt")
    
    print("\nSuccessful matches:")
    for entity, match in matches.items():
        print(f"{entity}: {match}")
    
    
    final_matches = {entity: match for entity, match in matches.items() if match.birthyear and match.birthplace and match.occupation}
    
    
    with open("src/prompt_entities_cleaned.txt", "w") as f:
        for entity in final_matches:
            f.write(f"{entity}\n")
    
    import pickle as pkl
    
    with open("src/matches.pkl", "wb") as f:
        pkl.dump(final_matches, f)
    
    print("\nAttribute coverage:")
    matched_attrs = analyze_wikidata_entities(matches)
    for attr, percentage in matched_attrs.items():
        print(f"{attr}: {percentage:.1f}%")