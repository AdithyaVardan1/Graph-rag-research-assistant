"""
Knowledge Graph construction and querying for GraphRAG.
Extracts entities and relationships from papers using LLM.

This is the core of the GraphRAG implementation - it transforms
unstructured paper content into a structured knowledge graph that
can be traversed for more intelligent retrieval.
"""
import json
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from core.llm_client import generate_completion

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity extracted from a paper."""
    name: str
    entity_type: str  # "CONCEPT", "METHOD", "ALGORITHM", "DATASET", "METRIC"
    paper_id: str
    
    def __hash__(self):
        return hash((self.name.lower(), self.entity_type))
    
    def __eq__(self, other):
        return self.name.lower() == other.name.lower() and self.entity_type == other.entity_type


@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    source: str
    target: str
    relation_type: str  # "USES", "IMPROVES", "COMPARES_TO", "BASED_ON", "APPLIES_TO"
    paper_id: str


class KnowledgeGraph:
    """
    Knowledge Graph for GraphRAG.
    
    Extracts entities and relationships from papers and provides
    graph-based retrieval capabilities.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}  # name.lower() -> Entity
        self.relationships: List[Relationship] = []
        self.entity_to_papers: Dict[str, Set[str]] = {}  # entity name -> set of paper IDs
        self.paper_entities: Dict[str, List[str]] = {}  # paper ID -> list of entity names
    
    def extract_from_paper(self, paper: Dict) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from a paper using LLM.
        
        Args:
            paper: Paper dict with 'id', 'title', 'summary' keys
            
        Returns:
            Tuple of (entities, relationships) extracted from the paper
        """
        prompt = f"""Analyze this research paper and extract key entities and their relationships.

PAPER TITLE: {paper.get('title', 'Unknown')}

PAPER ABSTRACT: {paper.get('summary', '')[:2000]}

Extract entities and relationships in the following JSON format:
{{
    "entities": [
        {{"name": "exact entity name", "type": "CONCEPT|METHOD|ALGORITHM|DATASET|METRIC"}}
    ],
    "relationships": [
        {{"source": "entity1 name", "relation": "USES|IMPROVES|COMPARES_TO|BASED_ON|APPLIES_TO", "target": "entity2 name"}}
    ]
}}

GUIDELINES:
- Focus on technical concepts, methods, algorithms, datasets, and metrics
- Entity names should be specific (e.g., "transformer architecture" not just "model")
- Include 5-15 key entities
- Include meaningful relationships between entities
- Return ONLY valid JSON, no other text

JSON:"""

        try:
            response = generate_completion(prompt)
            
            # Try to extract JSON from response
            json_str = response.strip()
            
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            entities = []
            for e in data.get('entities', []):
                if 'name' in e and 'type' in e:
                    entities.append(Entity(
                        name=e['name'],
                        entity_type=e['type'],
                        paper_id=paper['id']
                    ))
            
            relationships = []
            for r in data.get('relationships', []):
                if all(k in r for k in ['source', 'relation', 'target']):
                    relationships.append(Relationship(
                        source=r['source'],
                        target=r['target'],
                        relation_type=r['relation'],
                        paper_id=paper['id']
                    ))
            
            logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relations from: {paper['title'][:50]}")
            return entities, relationships
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return [], []
    
    def add_paper(self, paper: Dict):
        """
        Process a paper and add its entities/relationships to the graph.
        
        Args:
            paper: Paper dict with 'id', 'title', 'summary' keys
        """
        entities, relationships = self.extract_from_paper(paper)
        paper_id = paper['id']
        
        self.paper_entities[paper_id] = []
        
        for entity in entities:
            key = entity.name.lower()
            
            # Store or update entity
            if key not in self.entities:
                self.entities[key] = entity
            
            # Track which papers mention this entity
            if key not in self.entity_to_papers:
                self.entity_to_papers[key] = set()
            self.entity_to_papers[key].add(paper_id)
            
            # Track which entities are in this paper
            self.paper_entities[paper_id].append(key)
        
        self.relationships.extend(relationships)
    
    def query_related_entities(self, query: str, depth: int = 2) -> List[str]:
        """
        Find entities related to query terms via graph traversal.
        
        Uses BFS to traverse relationships up to specified depth.
        
        Args:
            query: User query string
            depth: How many hops to traverse in the graph
            
        Returns:
            List of related entity names
        """
        query_terms = set(query.lower().split())
        
        # Find matching entities (fuzzy match on query terms)
        matched = set()
        for term in query_terms:
            if len(term) < 3:  # Skip very short terms
                continue
            for entity_name in self.entities:
                if term in entity_name or entity_name in term:
                    matched.add(entity_name)
        
        # BFS traversal to find related entities
        related = set(matched)
        frontier = set(matched)
        
        for _ in range(depth):
            new_frontier = set()
            for rel in self.relationships:
                source = rel.source.lower()
                target = rel.target.lower()
                
                if source in frontier and target in self.entities:
                    if target not in related:
                        new_frontier.add(target)
                        related.add(target)
                
                if target in frontier and source in self.entities:
                    if source not in related:
                        new_frontier.add(source)
                        related.add(source)
            
            frontier = new_frontier
            if not frontier:
                break
        
        return list(related)
    
    def get_papers_for_entities(self, entity_names: List[str]) -> List[str]:
        """
        Get all paper IDs that mention any of the given entities.
        
        Args:
            entity_names: List of entity names to look up
            
        Returns:
            List of paper IDs (deduplicated)
        """
        paper_ids = set()
        for entity in entity_names:
            key = entity.lower()
            if key in self.entity_to_papers:
                paper_ids.update(self.entity_to_papers[key])
        return list(paper_ids)
    
    def get_entity_context(self, entity_name: str) -> Dict:
        """
        Get context about an entity including its relationships.
        
        Args:
            entity_name: Name of entity to get context for
            
        Returns:
            Dict with entity info and its relationships
        """
        key = entity_name.lower()
        if key not in self.entities:
            return {}
        
        entity = self.entities[key]
        
        # Find all relationships involving this entity
        related_rels = []
        for rel in self.relationships:
            if rel.source.lower() == key or rel.target.lower() == key:
                related_rels.append({
                    "source": rel.source,
                    "relation": rel.relation_type,
                    "target": rel.target
                })
        
        return {
            "name": entity.name,
            "type": entity.entity_type,
            "papers": list(self.entity_to_papers.get(key, [])),
            "relationships": related_rels
        }
    
    def to_visualization(self) -> Dict:
        """
        Export graph for frontend visualization.
        
        Returns:
            Dict with 'nodes' and 'edges' for Cytoscape.js
        """
        nodes = []
        for key, entity in self.entities.items():
            nodes.append({
                "id": key,
                "label": entity.name,
                "type": entity.entity_type,
                "paper_count": len(self.entity_to_papers.get(key, []))
            })
        
        # Deduplicate edges
        seen_edges = set()
        edges = []
        for rel in self.relationships:
            source_key = rel.source.lower()
            target_key = rel.target.lower()
            
            # Only include edges between entities we have
            if source_key in self.entities and target_key in self.entities:
                edge_key = (source_key, target_key, rel.relation_type)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": source_key,
                        "target": target_key,
                        "label": rel.relation_type
                    })
        
        return {"nodes": nodes, "edges": edges}
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return {
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "paper_count": len(self.paper_entities),
            "entity_types": list(set(e.entity_type for e in self.entities.values()))
        }
