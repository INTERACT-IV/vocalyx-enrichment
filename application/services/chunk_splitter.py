"""
Service de découpage intelligent des transcriptions en chunks pour l'enrichissement
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger("vocalyx")


class ChunkSplitter:
    """Découpe les transcriptions en chunks optimaux pour le traitement LLM"""
    
    def __init__(self, max_chunk_size: int = 500, max_duration: float = 60.0):
        """
        Initialise le découpeur de chunks.
        
        Args:
            max_chunk_size: Taille maximale d'un chunk en caractères
            max_duration: Durée maximale d'un chunk en secondes
        """
        self.max_chunk_size = max_chunk_size
        self.max_duration = max_duration
    
    def split_by_size(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Découpe les segments par taille maximale (caractères ou durée).
        
        Args:
            segments: Liste de segments de transcription
            
        Returns:
            Liste de chunks (chacun est une liste de segments)
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        current_duration = 0.0
        
        for seg in segments:
            seg_text = seg.get('text', '')
            seg_size = len(seg_text)
            seg_duration = seg.get('end', 0.0) - seg.get('start', 0.0)
            
            # Vérifier si on doit créer un nouveau chunk
            if (current_size + seg_size > self.max_chunk_size or 
                current_duration + seg_duration > self.max_duration):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [seg]
                current_size = seg_size
                current_duration = seg_duration
            else:
                current_chunk.append(seg)
                current_size += seg_size
                current_duration += seg_duration
        
        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"📦 Split into {len(chunks)} chunks (max size: {self.max_chunk_size} chars, max duration: {self.max_duration}s)")
        return chunks
    
    def split_by_speaker(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Découpe les segments par locuteur pour préserver le contexte par speaker.
        
        Args:
            segments: Liste de segments de transcription avec champ 'speaker'
            
        Returns:
            Liste de chunks groupés par speaker
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk = []
        current_speaker = None
        current_size = 0
        
        for seg in segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            seg_text = seg.get('text', '')
            seg_size = len(seg_text)
            
            # Si le speaker change et qu'on a déjà un chunk, le sauvegarder
            if speaker != current_speaker and current_chunk:
                # Si le chunk actuel est trop grand, le diviser
                if current_size > self.max_chunk_size:
                    sub_chunks = self._split_chunk_by_size(current_chunk)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(current_chunk)
                current_chunk = [seg]
                current_size = seg_size
            else:
                # Vérifier si on dépasse la taille max
                if current_size + seg_size > self.max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = [seg]
                    current_size = seg_size
                else:
                    current_chunk.append(seg)
                    current_size += seg_size
            
            current_speaker = speaker
        
        # Ajouter le dernier chunk
        if current_chunk:
            if current_size > self.max_chunk_size:
                sub_chunks = self._split_chunk_by_size(current_chunk)
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk)
        
        logger.info(f"🎤 Split into {len(chunks)} chunks by speaker")
        return chunks
    
    def split_by_semantic_boundaries(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Découpe aux limites sémantiques (phrases, pauses).
        
        Args:
            segments: Liste de segments de transcription
            
        Returns:
            Liste de chunks respectant les limites sémantiques
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for seg in segments:
            text = seg.get('text', '').strip()
            seg_size = len(text)
            
            # Détecter les limites sémantiques (point, point d'interrogation, pause)
            is_boundary = (
                text.endswith(('.', '?', '!')) or
                seg.get('pause_after', 0) > 1.0  # Pause de plus d'1 seconde
            )
            
            if is_boundary:
                current_chunk.append(seg)
                current_size += seg_size
                
                # Si on atteint la taille max ou qu'on a une bonne limite, créer un chunk
                if current_size >= self.max_chunk_size * 0.8:  # 80% de la taille max
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk.append(seg)
                current_size += seg_size
                
                # Si on dépasse la taille max, forcer la création d'un chunk
                if current_size > self.max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
        
        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"🧠 Split into {len(chunks)} chunks by semantic boundaries")
        return chunks
    
    def _split_chunk_by_size(self, chunk: List[Dict]) -> List[List[Dict]]:
        """Divise un chunk trop grand en plusieurs chunks plus petits"""
        sub_chunks = []
        current_sub_chunk = []
        current_size = 0
        
        for seg in chunk:
            seg_size = len(seg.get('text', ''))
            
            if current_size + seg_size > self.max_chunk_size:
                if current_sub_chunk:
                    sub_chunks.append(current_sub_chunk)
                current_sub_chunk = [seg]
                current_size = seg_size
            else:
                current_sub_chunk.append(seg)
                current_size += seg_size
        
        if current_sub_chunk:
            sub_chunks.append(current_sub_chunk)
        
        return sub_chunks
    
    def split(self, segments: List[Dict], strategy: str = 'size', use_diarization: bool = False) -> List[List[Dict]]:
        """
        Découpe les segments selon la stratégie choisie.
        
        Args:
            segments: Liste de segments de transcription
            strategy: Stratégie de découpage ('size', 'speaker', 'semantic')
            use_diarization: Si True, priorise le découpage par speaker
            
        Returns:
            Liste de chunks
        """
        if not segments:
            return []
        
        # Si la diarisation est disponible, utiliser le découpage par speaker
        if use_diarization and any(seg.get('speaker') for seg in segments):
            return self.split_by_speaker(segments)
        
        # Sinon, utiliser la stratégie demandée
        if strategy == 'speaker':
            return self.split_by_speaker(segments)
        elif strategy == 'semantic':
            return self.split_by_semantic_boundaries(segments)
        else:  # 'size' par défaut
            return self.split_by_size(segments)
