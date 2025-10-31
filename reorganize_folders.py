import os
import shutil
import pydicom
import pydicom.dataelem
import argparse
import logging
import sys
import json
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any, Protocol, Tuple
from functools import wraps, lru_cache
from abc import ABC, abstractmethod
from performance_profiler import profiler, measure, setup_profiling, measure_block
import mmap
from itertools import islice
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing as mp
from threading import Lock
import queue

# Global error tracking for suppressing repetitive warnings
ERROR_COUNTS = defaultdict(int)
ERROR_SUPPRESSION_THRESHOLD = 5

# Global patient scan cache to prevent re-scanning
PATIENT_SCAN_CACHE = {}


# --- Data Classes ---
@dataclass
class SeriesInfo:
    uid: str
    files: List[str]
    series_number: int
    study_datetime: datetime
    first_dataset: pydicom.Dataset
    protocol_name: str = ""
    series_desc: str = ""


@dataclass
class StudyInfo:
    uid: str
    series: Dict[str, SeriesInfo]
    study_datetime: datetime


@dataclass
class PatientData:
    original_id: str
    studies: Dict[str, StudyInfo]


@dataclass
class ModalitySelectionLog:
    """Enhanced logging for modality selection decisions."""
    session_id: str
    modality: str
    selected_protocol: str
    selection_reason: str
    strategy_used: str
    candidates_considered: List[str]
    forbidden_filtered: List[str]
    priority_scores: Dict[str, float]
    year_detected: Optional[int] = None


# --- Enhanced Protocol Detection Strategy Pattern ---
class ModalityDetectionStrategy(ABC):
    """Abstract base class for modality detection strategies."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this detection strategy."""
        pass
    
    @abstractmethod
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Check if this strategy is applicable to the given dataset."""
        pass
    
    @abstractmethod
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality using this strategy. Returns None if not detected."""
        pass
    
    def get_priority(self) -> int:
        """Return priority for this strategy (lower number = higher priority)."""
        return 100
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        return []
    
    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        """Return scoring weights for priority selection."""
        return {}
    
    def is_exclusive(self) -> bool:
        """Return True if this strategy should prevent fallback to other strategies when applicable."""
        return False


class EnhancedStandardDetectionStrategy(ModalityDetectionStrategy):
    """Enhanced standard modality detection strategy based on CSV analysis."""
    
    def __init__(self):
        # Enhanced dictionaries based on CSV analysis
        self.modality_config = {
            't1c': {
                'keywords': ['t1', 'ce'],  # Both must be present
                'alt_keywords': ['t1', 'contrast', 'gad', 'postcontrast', '+c', 'post', 'c+', 'c', 'gd'],
                'forbidden': ['mpr', 'dyn', 'pit', 'spir'],
                'priority_sequences': ['tfe', 'tse', 'se'],
                'priority_modifiers': ['3d', 'brain'],
                'scoring_weights': {
                    'tfe': 3.0,
                    'tse': 2.0,
                    'se': 1.0,
                    '3d': 1.5,
                    'brain': 1.1,
                    'axi': 1.3,
                    'sag': 1.2,
                    'cor': 1.1,
                    'thr': 0.8,
                    'mpr': 0.1
                }
            },
            't1': {
                'keywords': ['t1', 't1w'],
                'forbidden': ['thr', 'ce', 'pit', 'contrast', 'gad', 'c+'], # временно убран 'mpr' для корректной обработки UPENN
                'priority_sequences': ['tfe', 'tse', 'se'],
                'priority_modifiers': ['3d', 'clear', 'brain'],
                'scoring_weights': {
                    'tfe': 3.0,
                    'tse': 2.0, 
                    'se': 1.0,
                    '3d': 1.5,
                    'clear': 1.2,
                    'brain': 1.1,
                    'ax': 1.3,
                    'sag': 1.2,
                    'cor': 1.1,
                    'mpr': 0.1
                }
            },
            't2': {
                'keywords': ['t2', 't2w'],
                'forbidden': ['pit', 'mpr', 'contrast', 'flair'], # временно убран 'ce' из-за 't2 sag space'
                'priority_sequences': ['t2','tse'],
                'priority_modifiers': ['sense', 'brain', 'axi'],
                'scoring_weights': {
                    't2': 3.0,
                    'tse': 2.0,
                    'sense': 1.3,
                    'brain': 1.1,
                    'axi': 1.3,
                    'sag': 1.2,
                    'cor': 1.1,
                    'st2': 0.8,
                    'mpr': 0.1
                }
            },
            't2fl': {
                'keywords': ['flair'],
                'forbidden': ['mpr', 'ce', 'spir', 'contrast'],
                'priority_sequences': [],
                'priority_modifiers': ['3d', 'sense', 'long', 'brain'],
                'scoring_weights': {
                    '3d': 2.0,
                    'sense': 1.3,
                    'long': 1.2,
                    'view': 0.1,
                    'mpr': 0.1
                }
            }
        }
    
    def get_name(self) -> str:
        return "Enhanced Standard Detection"
    
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Enhanced standard detection is always applicable as fallback."""
        return True
    
    def get_priority(self) -> int:
        return 1000  # Lowest priority (highest number)
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        config = self.modality_config.get(modality, {})
        return config.get('priority_sequences', []) + config.get('priority_modifiers', [])
    
    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        """Return scoring weights for priority selection."""
        config = self.modality_config.get(modality, {})
        return config.get('scoring_weights', {})
    
    def is_exclusive(self) -> bool:
        """Enhanced standard detection is never exclusive - it's the fallback."""
        return False
    
    @measure
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality using enhanced workflow."""
        logger = logging.getLogger(__name__)
        
        # Try different detection methods in order
        detectors = [
            ("Enhanced Protocol Analysis", self._detect_by_enhanced_protocol),
            ("Series Description Analysis", self._detect_by_enhanced_series_description),
            #("Technical Parameters", self._detect_by_technical_params), - временно убран, потому что определяет 'ep2d_dti_30dir' как t2, что неправильно
            ("File Path Analysis", lambda ds, fp: self._detect_by_file_path(fp))
        ]
        
        for detector_name, detector_func in detectors:
            logger.debug(f"  Trying {detector_name}...")
            modality = detector_func(ds, file_path)
            if modality:
                logger.debug(f"  ✓ Detected '{modality}' using {detector_name}")
                return modality
            else:
                logger.debug(f"  ✗ No match using {detector_name}")
        
        return None
    
    def _detect_by_enhanced_protocol(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Enhanced protocol detection with improved keyword matching."""
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        
        has_contrast = self._check_contrast_presence(ds, protocol_name, contrast_agent)
        
        logger = logging.getLogger(__name__)
        logger.debug(f"    Protocol: '{protocol_name}', Contrast: {has_contrast}")
        
        return self._analyze_text_for_modality(protocol_name, has_contrast)
    
    def _detect_by_enhanced_series_description(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Enhanced series description detection."""
        series_desc = normalize_dicom_text(get_dicom_value(ds, (0x0008, 0x103E), ""))
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        
        has_contrast = self._check_contrast_presence(ds, series_desc, contrast_agent)
        
        logger = logging.getLogger(__name__)
        logger.debug(f"    Series Desc: '{series_desc}', Contrast: {has_contrast}")
        
        return self._analyze_text_for_modality(series_desc, has_contrast)
    
    def _check_contrast_presence(self, ds: pydicom.Dataset, text: str, contrast_agent: str) -> bool:
        """Enhanced contrast detection."""
        # Check contrast agent field
        if contrast_agent and contrast_agent not in ["", "none", "no"]:
            return True
        
        if 'c+t1' in text.lower() or 'c+' in text.lower():
            return True
        
        # Check for contrast keywords in text
        contrast_keywords = ['ce', 'contrast', 'gad', 'gadolinium', 'post', '+c', 'enhanced', 'c+']
        if any(keyword in text.lower() for keyword in contrast_keywords):
            return True
        
        # Check additional DICOM fields
        contrast_bolusagent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1078), ""))
        if contrast_bolusagent and contrast_bolusagent not in ["", "none", "no"]:
            return True
        
        return False
    
    @measure
    def _analyze_text_for_modality(self, text: str, has_contrast: bool) -> Optional[str]:
        """Analyze text for modality with enhanced logic."""
        if not text:
            return None
        
        logger = logging.getLogger(__name__)
        
        # First check forbidden words for each modality
        candidates = []
        for modality, config in self.modality_config.items():
            # Check forbidden words
            if any(forbidden in text.lower() for forbidden in config.get('forbidden', [])):
                logger.debug(f"      {modality} rejected - forbidden word found")
                continue
            
            # Check required keywords
            if self._check_keywords_match(text, modality, config):
                # Special handling for contrast-dependent modalities
                if modality == 't1c':
                    if has_contrast:
                        candidates.append(modality)
                        logger.debug(f"      {modality} candidate - keywords + contrast")
                elif modality in ['t1', 't2', 't2fl']:
                    # For non-contrast modalities, prefer if no contrast detected
                    if not has_contrast or modality == 't2fl':  # FLAIR can be post-contrast sometimes
                        candidates.append(modality)
                        logger.debug(f"      {modality} candidate - keywords, no contrast conflict")
        
        # If multiple candidates, prioritize based on specificity
        if len(candidates) > 1:
            # FLAIR is most specific
            if 't2fl' in candidates:
                return 't2fl'
            # T1C is more specific than T1
            if 't1c' in candidates and has_contrast:
                return 't1c'
            if 't1' in candidates and not has_contrast:
                return 't1'
            # Return first candidate
            return candidates[0]
        elif len(candidates) == 1:
            return candidates[0]
        
        return None
    
    def _check_keywords_match(self, text: str, modality: str, config: Dict) -> bool:
        """Check if text matches keywords for a modality."""
        text_lower = text.lower()
        
        # Primary keywords (all must be present for t1c)
        keywords = config.get('keywords', [])
        if modality == 't1c':
            if 'c+t1' in text_lower:
                return True
            
            if 't1+c' in text_lower or 't1c+' in text_lower:
                return True
            # For T1C, require both 't1' and 'ce' (or check alternative keywords)
            primary_match = all(keyword in text_lower for keyword in keywords)
            
            # Check alternative contrast keywords if primary doesn't match
            if not primary_match:
                alt_keywords = config.get('alt_keywords', [])
                if alt_keywords:
                    # Need 't1' plus any contrast indicator
                    has_t1 = 't1' in text_lower
                    has_contrast_kw = any(kw in text_lower for kw in alt_keywords[1:])  # Skip 't1'
                    primary_match = has_t1 and has_contrast_kw
            
            return primary_match
        else:
            # For other modalities, any keyword match is sufficient
            return any(keyword in text_lower for keyword in keywords)
    
    def _detect_by_technical_params(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Enhanced technical parameter detection."""
        tr_val = safe_float(get_dicom_value(ds, (0x0018, 0x0080)), "TR")
        te_val = safe_float(get_dicom_value(ds, (0x0018, 0x0081)), "TE")
        ti_val = safe_float(get_dicom_value(ds, (0x0018, 0x0082)), "TI")
        
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        has_contrast = self._check_contrast_presence(ds, "", contrast_agent)
        
        logger = logging.getLogger(__name__)
        logger.debug(f"    TR={tr_val}, TE={te_val}, TI={ti_val}, Contrast={has_contrast}")
        
        # FLAIR detection by TI (most specific)
        if ti_val and ti_val > 1500:
            logger.debug(f"    Technical params suggest FLAIR (TI={ti_val})")
            return 't2fl'
        
        # T1C detection by contrast + T1 parameters
        if has_contrast and tr_val and te_val and tr_val < 1200 and te_val < 30:
            logger.debug(f"    Technical params suggest T1C (contrast + TR/TE)")
            return 't1c'
        
        # T1 detection by TR/TE without contrast
        if tr_val and te_val and tr_val < 1000 and te_val < 30 and not has_contrast:
            logger.debug(f"    Technical params suggest T1 (TR/TE, no contrast)")
            return 't1'
        
        # T2 detection by TR/TE
        if tr_val and te_val and tr_val > 2000 and te_val > 70:
            if not ti_val or ti_val < 1500:  # Make sure it's not FLAIR
                logger.debug(f"    Technical params suggest T2 (TR/TE, not FLAIR)")
                return 't2'
        
        return None
    
    def _detect_by_file_path(self, file_path: str) -> Optional[str]:
        """Enhanced file path analysis."""
        path_parts = os.path.normpath(file_path).split(os.sep)
        for part in reversed(path_parts):
            part_lower = part.lower()
            
            modality = self._analyze_text_for_modality(part_lower, 'contrast' in part_lower)
            if modality:
                return modality
        
        return None


class YearSpecificDetectionStrategy(ModalityDetectionStrategy):
    """Year-specific detection strategy based on CSV analysis."""
    
    def __init__(self, target_years: List[int], strategy_name: str, keywords_config: Dict):
        self.target_years = target_years
        self.strategy_name = strategy_name
        self.keywords_config = keywords_config
    
    def get_name(self) -> str:
        years_str = ", ".join(map(str, self.target_years))
        return f"{self.strategy_name} ({years_str})"
    
    def get_priority(self) -> int:
        return 10  # Higher priority than standard
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        config = self.keywords_config.get(modality, {})
        return config.get('prefer_order', [])
    
    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        """Return scoring weights for priority selection."""
        config = self.keywords_config.get(modality, {})
        return config.get('scoring_weights', {})
    
    def is_exclusive(self) -> bool:
        """Year-specific protocols are exclusive - if year matches, don't fall back."""
        return True
    
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Check if this is a study from target years."""
        study_date_val = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date_val, str) and len(study_date_val) >= 4:
            try:
                year = int(study_date_val[:4])
                return year in self.target_years
            except ValueError:
                pass
        return False
    
    @measure
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality using year-specific rules."""
        logger = logging.getLogger(__name__)
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        series_desc = normalize_dicom_text(get_dicom_value(ds, (0x0008, 0x103E), ""))
        
        # Combine protocol name and series description for analysis
        combined_text = f"{protocol_name} {series_desc}".strip()
        
        logger.debug(f"  Year-specific analysis: '{combined_text}'")
        
        for modality, rules in self.keywords_config.items():
            if self._matches_year_specific_rules(combined_text, rules):
                logger.debug(f"  ✓ Matched modality '{modality}' with year-specific rules")
                return modality
        
        logger.debug("  ✗ No match found with year-specific rules")
        return None
    
    def _matches_year_specific_rules(self, text: str, rules: Dict) -> bool:
        """Check if text matches year-specific rules."""
        text_lower = text.lower()
        
        # Check forbidden keywords first
        forbidden = rules.get('forbidden', [])
        if any(fw in text_lower for fw in forbidden):
            return False
        
        # Check required keywords
        required = rules.get('required', [])
        if required:
            # Проверяем основное условие: все required keywords присутствуют
            all_required_found = all(rw in text_lower for rw in required)
            
            if not all_required_found:
                # Если основное условие не выполнено, проверяем альтернативы
                alt_keywords = rules.get('alt_keywords', [])
                if alt_keywords:
                    # Для модальности t1c: специальная логика для контраста
                    # Проверяем наличие базового keyword (обычно 't1') и любого из альтернативных
                    if 't1' in required:  # Это модальность с контрастом
                        has_base = 't1' in text_lower
                        # Проверяем альтернативные индикаторы контраста
                        contrast_indicators = [kw for kw in alt_keywords if kw != 't1']
                        has_contrast = any(kw in text_lower for kw in contrast_indicators)
                        
                        if not (has_base and has_contrast):
                            return False
                    else:
                        # Для других модальностей: любой из alt_keywords должен присутствовать
                        if not any(kw in text_lower for kw in alt_keywords):
                            return False
                else:
                    # Нет альтернатив и required не найдены
                    return False
        
        # Check marker keywords
        markers = rules.get('markers', [])
        if markers and not any(mk in text_lower for mk in markers):
            return False
        
        return True


# --- Enhanced Modality Detector ---
class EnhancedModalityDetector:
    """Enhanced modality detector with comprehensive year-specific strategies."""
    
    def __init__(self):
        # Initialize all available strategies
        self.strategies: List[ModalityDetectionStrategy] = []
        
        # Add year-specific strategies based on CSV analysis
        self._add_year_specific_strategies()
        
        # Add enhanced standard strategy as fallback
        self.strategies.append(EnhancedStandardDetectionStrategy())
        
        # Sort by priority (lower number = higher priority)
        self.strategies.sort(key=lambda s: s.get_priority())
        self.logger = logging.getLogger(__name__)
        
        # Initialize selection logging
        self.selection_logs: List[ModalitySelectionLog] = []
    
    def _add_year_specific_strategies(self):
        """Add year-specific strategies based on CSV analysis."""
        
        # 2018 Strategy
        strategy_2018 = YearSpecificDetectionStrategy(
            target_years=[2018],
            strategy_name="Protocol 2018",
            keywords_config={
                't1c': {
                    'required': ['ce', 't1'],
                    'forbidden': [],
                    'prefer_order': ['se'],
                    'scoring_weights': {'se': 2.0}
                },
                't1': {
                    'markers': ['t1w', 't1'],
                    'forbidden': ['thr', 'ce'],
                    'prefer_order': ['tfe', 'tse', 'se'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, 'se': 1.0}
                },            
                't2': {
                    'markers': ['t2w', 't2'],
                    'forbidden': ['flair'],
                    'prefer_order': ['tse','tra','st2'],
                    'scoring_weights': {'tse': 2.0, 'tra': 1.5, 'st2': 0.5}
                },
                't2fl': {
                    'markers': ['flair'],
                    'forbidden': [],
                    'prefer_order': ['sense', 'long'],
                    'scoring_weights': {'sense': 1.5, 'long': 1.2}
                }
            }
        )
        
        # 2020 Strategy
        strategy_2020 = YearSpecificDetectionStrategy(
            target_years=[2020],
            strategy_name="Protocol 2020",
            keywords_config={
                't1c': {
                    'required': ['ce', 't1'],
                    'alt_keywords': ['t1', 'contrast', 'gad', 'postcontrast', '+c', 'post', 'c+', '_c', 'gd'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['tfe', 'tse', '3d', 'se'],
                    'scoring_weights': {'tfe': 3.0,'tse': 2.0, '3d': 1.5, 'se': 1.0, 'clear': 1.5, 'ax': 1.3, 'sag': 1.2, 'cor': 1.1}
                },
                't1': {
                    'markers': ['t1w', 't1-tse', 't1'],
                    'forbidden': ['thr', 'mpr', 'ce'],
                    'prefer_order': ['tfe', 'tse', 'clear', '3d', 'se'],
                    'scoring_weights': {'tfe' : 2.5,'tse': 2.0, 'clear': 1.5, '3d': 1.3, 'se': 1.1, 'ax': 1.3, 'sag': 1.2, 'cor': 1.1}
                },               
                't2': {
                    'markers': ['t2w', 't2-tse', 't2'],
                    'forbidden': ['mpr', 'flair'],
                    'prefer_order': ['tse', 'clear', 'sense', 'axi'],
                    'scoring_weights': {'tse': 2.0, 'clear': 1.4, 'sense': 1.3, 'axi': 1.3, 'sag': 1.2, 'cor': 1.1, 'st2': 0.9}
                },
                't2fl': {
                    'markers': ['flair'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['3d', 'long', 'sense'],
                    'scoring_weights': {'3d': 2.0, 'long': 1.2, 'sense': 1.1}
                }
            }
        )
        
        # 2021-2022 Strategy
        strategy_2021_2022 = YearSpecificDetectionStrategy(
            target_years=[2021, 2022],
            strategy_name="Protocol 2021-2022",
            keywords_config={
                't1c': {
                    'required': ['ce', 't1'], # как-то надо учесть gd, +c 'st1w_3d_iso+c st1w_3d_iso+c'
                    'alt_keywords': ['t1', 'contrast', 'gad', 'postcontrast', '+c', 'post', 'c+', '_c', 'gd'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['tfe', 'tse', '3d'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5, 'delay': 0.8}
                },
                't1': {
                    'markers': ['t1-tfe', 't1-tse', 't1w', 't1'],
                    'forbidden': ['mpr', 'ce', 'gd'],
                    'prefer_order': ['tfe', 'tse', '3d'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}
                },

                't2': {
                    'markers': ['t2-tse', 't2_tse', 't2', 't2_ffe'],
                    'forbidden': ['mpr', 'flair'],
                    'prefer_order': ['tse', 'axi'],
                    'scoring_weights': {'tse': 2.0, 'axi': 1.2}
                },
                't2fl': {
                    'markers': ['flair'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['3d'],
                    'scoring_weights': {'3d': 2.0}
                }
            }
        )
        
        # 2023+ Strategy
        strategy_2023_plus = YearSpecificDetectionStrategy(
            target_years=[2023, 2024, 2025],
            strategy_name="Protocol 2023+",
            keywords_config={
                't1c': {
                    'required': ['ce', 't1'], # C+
                    'alt_keywords': ['t1', 'contrast', 'gad', 'postcontrast', '+c', 'post', 'c+', '_c', 'gd'],
                    'forbidden': ['mpr', 'dyn', 'pit'],
                    'prefer_order': ['tfe', 'tse', '3d'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}
                },
                't1': {
                    'markers': ['t1-tfe', 't1-tse', 't1'],
                    'forbidden': ['mpr', 'ce'],
                    'prefer_order': ['tfe', 'tse', '3d'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}
                },

                't2': {
                    'markers': ['t2-tse', 't2_tse', 't2'],
                    'forbidden': ['pit', 'mpr', 'flair'],
                    'prefer_order': ['tse', 'axi', 'ce'],
                    'scoring_weights': {'tse': 2.0, 'axi': 1.2, 'ce': 0.5}
                },
                't2fl': {
                    'markers': ['flair'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['3d', 'ce'],
                    'scoring_weights': {'3d': 2.0, 'ce': 0.5}
                }
            }
        )
        
        self.strategies.extend([
            strategy_2018,
            strategy_2020,
            strategy_2021_2022,
            strategy_2023_plus
        ])
    
    def add_strategy(self, strategy: ModalityDetectionStrategy):
        """Add a new detection strategy."""
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.get_priority())
        self.logger.info(f"Added detection strategy: {strategy.get_name()}")
    
    @measure(capture_args=True)
    def determine_modality_with_details(self, ds: pydicom.Dataset, file_path: str, session_id: str = "") -> Tuple[str, Optional[ModalityDetectionStrategy], Dict]:
        """Determine modality with detailed logging information."""
        self.logger.debug(f"Determining modality for {os.path.basename(file_path)}:")
        
        details = {
            'candidates_considered': [],
            'forbidden_filtered': [],
            'strategy_used': None,
            'year_detected': None
        }
        
        # Extract year for logging
        study_date_val = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date_val, str) and len(study_date_val) >= 4:
            try:
                details['year_detected'] = int(study_date_val[:4])
            except ValueError:
                pass
        
        # Try each strategy in priority order
        for strategy in self.strategies:
            strategy_name = strategy.get_name()
            
            # Check if strategy is applicable
            if strategy.is_applicable(ds, file_path):
                self.logger.debug(f"→ Trying strategy: {strategy_name}")
                details['strategy_used'] = strategy_name
                
                # Try to detect modality
                modality = strategy.detect_modality(ds, file_path)
                
                if modality and modality != 'unknown':
                    self.logger.info(f"✓ Modality '{modality}' detected using strategy: {strategy_name}")
                    return modality, strategy, details
                else:
                    # If this is an exclusive strategy and it didn't match, skip the file entirely
                    if strategy.is_exclusive():
                        self.logger.warning(f"  Exclusive strategy {strategy_name} did not match - skipping file")
                        return 'unknown', None, details
                    else:
                        self.logger.debug(f"  Strategy {strategy_name} did not detect modality")
            else:
                self.logger.debug(f"  Strategy {strategy_name} is not applicable")
        
        self.logger.warning(f"Could not determine modality for file: {os.path.basename(file_path)}")
        return 'unknown', None, details
    
    @measure
    def determine_modality(self, ds: pydicom.Dataset, file_path: str) -> str:
        """Determine modality using registered strategies."""
        modality, _, _ = self.determine_modality_with_details(ds, file_path)
        return modality


# --- Global Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Error Handling Decorator ---
def handle_dicom_error(func):
    """Decorator for consistent DICOM error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


# --- Error Suppression Functions ---
def log_parsing_error_once(error_type: str, sample_uid: str, logger_instance):
    """Log parsing errors only once with counting to reduce log spam."""
    global ERROR_COUNTS, ERROR_SUPPRESSION_THRESHOLD
    
    ERROR_COUNTS[error_type] += 1
    
    # Log only first few errors of each type
    if ERROR_COUNTS[error_type] <= ERROR_SUPPRESSION_THRESHOLD:
        logger_instance.warning(f"Parsing error ({error_type}): {sample_uid}")
    elif ERROR_COUNTS[error_type] == ERROR_SUPPRESSION_THRESHOLD + 1:
        logger_instance.warning(f"Suppressing further '{error_type}' errors (total so far: {ERROR_COUNTS[error_type]})")

def get_error_suppression_summary():
    """Get summary of suppressed errors."""
    global ERROR_COUNTS
    return dict(ERROR_COUNTS)

# --- Utility Functions ---
def setup_logging(log_file_path: str):
    """Set up logging to console (INFO) and file (DEBUG)."""
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"File logging configured: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to configure file logging {log_file_path}: {e}")

@measure
def normalize_dicom_text(value: Any) -> str:
    """Convert DICOM value to normalized lowercase string."""
    if value is None:
        return ""
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return " ".join(str(v).strip().lower() for v in value if v is not None)
    return str(value).strip().lower()


def safe_float(value: Any, tag_name: str = "value") -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.debug(f"Cannot convert {tag_name}='{value}' (type: {type(value)}) to float.")
        return None

@measure
def get_dicom_value(ds: pydicom.Dataset, tag: Union[tuple, str], default: Any = None) -> Any:
    """Safely extract value from DICOM dataset."""
    try:
        val = ds.get(tag, default)
        if val is default:
            return default
        
        if isinstance(val, pydicom.dataelem.DataElement):
            val = val.value
            if val is None:
                return default
        
        if val is None:
            return default
            
        if isinstance(val, str):
            return val.strip().lower()
        if isinstance(val, (pydicom.multival.MultiValue, list)):
            return [v.strip().lower() if isinstance(v, str) else v for v in val if v is not None]
        return val
    except Exception as e:
        logger.error(f"Exception in get_dicom_value for tag {tag}: {e}")
        return default

@measure(capture_args=True) 
@handle_dicom_error
def safe_read_dicom(file_path: str, specific_tags: Optional[List] = None) -> Optional[pydicom.Dataset]:
    """Safely read DICOM file with error handling."""
    return pydicom.dcmread(file_path, stop_before_pixels=True, specific_tags=specific_tags)

# PERFORMANCE FIX: Thread-local cache instead of global LRU cache to avoid GIL contention
import threading
_thread_local_cache = threading.local()

def _cached_dicom_header(file_path: str) -> Optional[pydicom.Dataset]:
    """
    Read DICOM header with thread-local caching to avoid GIL contention.
    Each thread maintains its own cache to prevent lock contention.
    """
    # Get thread-local cache
    if not hasattr(_thread_local_cache, 'cache'):
        _thread_local_cache.cache = {}
    
    cache = _thread_local_cache.cache
    
    # Check thread-local cache first
    if file_path in cache:
        return cache[file_path]
    
    # Read file without mmap to avoid filesystem locks
    tags = [
        (0x0008, 0x0060), (0x0008, 0x103E), (0x0018, 0x1030), (0x0018, 0x0010),
        (0x0008, 0x0020), (0x0008, 0x0030), (0x0020, 0x000D), (0x0020, 0x000E),
        (0x0020, 0x0011), (0x0020, 0x0013), (0x0018, 0x0020), (0x0018, 0x0021),
        (0x0008, 0x0008), (0x0018, 0x0080), (0x0018, 0x0081), (0x0018, 0x0082),
        (0x0010, 0x0020), (0x0008, 0x0022), (0x0008, 0x0032)
    ]
    
    try:
        # PERFORMANCE FIX: Direct file read without mmap to avoid filesystem contention
        dataset = pydicom.dcmread(file_path, stop_before_pixels=True, specific_tags=tags)
        
        # PERFORMANCE FIX: Increase cache size significantly for better hit rate
        if len(cache) < 10000:  # 10K files per thread = 200K total cache (20 threads × 10K)
            cache[file_path] = dataset
        
        return dataset
    except Exception as e:
        logger.debug(f"Failed to read DICOM header from {file_path}: {e}")
        return None

def clear_dicom_cache():
    """Clear the DICOM header cache to free memory."""
    # PERFORMANCE FIX: Clear thread-local caches instead of global LRU cache
    # Note: This only clears the current thread's cache
    if hasattr(_thread_local_cache, 'cache'):
        cache_size = len(_thread_local_cache.cache)
        _thread_local_cache.cache.clear()
        logger.info(f"Thread-local DICOM cache cleared ({cache_size} entries)")
    else:
        logger.info("No thread-local DICOM cache to clear")

def clear_patient_scan_cache():
    """Clear the patient scan cache to free memory."""
    global PATIENT_SCAN_CACHE
    cache_size = len(PATIENT_SCAN_CACHE)
    PATIENT_SCAN_CACHE.clear()
    logger.info(f"Patient scan cache cleared ({cache_size} entries)")

def get_cache_statistics():
    """Get current cache statistics for monitoring."""
    # PERFORMANCE FIX: Get thread-local cache stats instead of global LRU
    thread_cache_size = 0
    if hasattr(_thread_local_cache, 'cache'):
        thread_cache_size = len(_thread_local_cache.cache)
    
    patient_cache_size = len(PATIENT_SCAN_CACHE)
    error_counts = get_error_suppression_summary()
    
    return {
        'dicom_cache_hits': 'N/A (thread-local)',
        'dicom_cache_misses': 'N/A (thread-local)',
        'dicom_cache_size': thread_cache_size,
        'patient_cache_size': patient_cache_size,
        'total_errors_suppressed': sum(error_counts.values()),
        'error_types': len(error_counts)
    }


@measure 
def is_dicom_file(file_path: str) -> bool:
    """Check if file is a valid DICOM file."""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except pydicom.errors.InvalidDicomError:
        logger.debug(f"File {file_path} is not a valid DICOM.")
        return False
    except Exception as e:
        logger.warning(f"Error checking DICOM file {file_path}: {e}")
        return False


# --- DICOM Scanner Class ---
class DicomScanner:
    """Handles scanning and collecting DICOM files."""

    def __init__(self):
        """Добавляем инициализацию для параллельной обработки."""
        self.progress_lock = Lock()
        self.progress_counter = 0
        self.total_files = 0
        # PERFORMANCE FIX: Add patient-level caching to prevent re-scanning
        self.patient_scan_results = {}
        self.scan_start_time = None

    @measure(capture_args=True)
    def scan_directory(self, input_dir: str) -> Dict[str, PatientData]:
        """Scan directory and collect DICOM metadata with parallel processing."""
        logger.info("Phase 1: Scanning DICOM files and collecting metadata...")
        
        # PERFORMANCE FIX: Track scanning performance
        self.scan_start_time = time.time()
        
        # Начало фазы
        profiler.record_phase("phase_1_scanning", "start")
        profiler.memory_checkpoint("scan_start")
        
        collected_data = {}
        
        # PERFORMANCE FIX: Check if we can use cached results
        cache_key = f"scan_{os.path.abspath(input_dir)}"
        if cache_key in PATIENT_SCAN_CACHE:
            logger.info("  Using cached scan results from previous run")
            return PATIENT_SCAN_CACHE[cache_key]
        
        # Шаг 1: Собираем все файлы для обработки
        logger.info("  Collecting file list...")
        all_files = []
        for root, _, files in os.walk(input_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                all_files.append(file_path)
        
        self.total_files = len(all_files)
        logger.info(f"  Found {self.total_files} total files to process")
        
        # PERFORMANCE FIX: Log estimated processing time based on previous runs
        if self.total_files > 0:
            estimated_time = self.total_files / 40  # Based on observed ~40 files/sec
            logger.info(f"  Estimated processing time: {estimated_time:.1f} seconds")
        
        # Шаг 2: Параллельная обработка файлов
        logger.info("  Processing files in parallel...")
        
        # PERFORMANCE FIX: Use ALL 20 threads on server for maximum throughput
        max_workers = 20  # Use all available threads on 20-core server
        
        # PERFORMANCE FIX: Lock-free parallel processing with batch collection
        # Collect all results first, then merge without locks
        logger.info(f"  Using {max_workers} parallel workers for file processing")
        
        # Используем ThreadPoolExecutor для I/O-bound задач
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Отправляем задачи на обработку
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in all_files
            }
            
            # PERFORMANCE FIX: Collect results without locks, merge at end
            all_results = []
            processed_count = 0
            
            for future in as_completed(future_to_file):
                try:
                    result = future.result(timeout=30)  # 30 секунд таймаут на файл
                    if result:
                        all_results.append(result)
                    
                    processed_count += 1
                    # Log progress less frequently to reduce I/O overhead
                    if processed_count % 1000 == 0:
                        logger.info(f"    Processed {processed_count}/{self.total_files} files...")
                            
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Error processing file {file_path}: {e}")
        
        # PERFORMANCE FIX: Single-threaded merge of all results (no locks needed)
        logger.info(f"  Merging {len(all_results)} results into patient data structure...")
        for result in all_results:
            self._add_series_info_to_collected(collected_data, result)
        
        # PERFORMANCE FIX: Cache results and log performance metrics
        PATIENT_SCAN_CACHE[cache_key] = collected_data
        
        # Calculate and log performance metrics
        scan_duration = time.time() - self.scan_start_time
        files_per_second = processed_count / scan_duration if scan_duration > 0 else 0
        
        # Log error suppression summary
        error_summary = get_error_suppression_summary()
        if error_summary:
            total_suppressed = sum(error_summary.values())
            logger.info(f"  Error suppression summary: {total_suppressed} total errors")
            for error_type, count in error_summary.items():
                if count > ERROR_SUPPRESSION_THRESHOLD:
                    logger.info(f"    {error_type}: {count} occurrences (suppressed after {ERROR_SUPPRESSION_THRESHOLD})")
        
        # Конец фазы
        profiler.record_phase("phase_1_scanning", "end")
        profiler.memory_checkpoint("scan_end")
        
        logger.info(f"Phase 1 completed. Total files processed: {processed_count}.")
        logger.info(f"  Processing speed: {files_per_second:.1f} files/second")
        logger.info(f"  Total scan duration: {scan_duration:.1f} seconds")
        
        return collected_data
    
    @measure
    def _add_series_info_to_collected(self, collected_data: Dict, series_info: Dict):
        """Thread-safe добавление информации в collected_data."""
        if not series_info:
            return
        
        pat_id = series_info['pat_id']
        study_uid = series_info['study_uid']
        series_uid = series_info['series_uid']
        
        # PERFORMANCE FIX: Minimize lock contention by preparing data outside lock
        # Prepare new objects outside the lock
        new_patient = None
        new_study = None
        new_series = None
        
        # Check what needs to be created (read-only operations)
        needs_patient = pat_id not in collected_data
        needs_study = not needs_patient and study_uid not in collected_data[pat_id].studies
        needs_series = not needs_study and not needs_patient and series_uid not in collected_data[pat_id].studies[study_uid].series
        
        # Prepare objects outside lock
        if needs_patient:
            new_patient = PatientData(original_id=pat_id, studies={})
            new_study = StudyInfo(
                uid=study_uid,
                series={},
                study_datetime=series_info['study_datetime']
            )
            new_series = SeriesInfo(
                uid=series_uid,
                files=[series_info['file_path']],
                series_number=series_info['series_number'],
                study_datetime=series_info['study_datetime'],
                first_dataset=series_info['dataset'],
                protocol_name=series_info['protocol_name'],
                series_desc=series_info['series_desc']
            )
        elif needs_study:
            new_study = StudyInfo(
                uid=study_uid,
                series={},
                study_datetime=series_info['study_datetime']
            )
            new_series = SeriesInfo(
                uid=series_uid,
                files=[series_info['file_path']],
                series_number=series_info['series_number'],
                study_datetime=series_info['study_datetime'],
                first_dataset=series_info['dataset'],
                protocol_name=series_info['protocol_name'],
                series_desc=series_info['series_desc']
            )
        elif needs_series:
            new_series = SeriesInfo(
                uid=series_uid,
                files=[series_info['file_path']],
                series_number=series_info['series_number'],
                study_datetime=series_info['study_datetime'],
                first_dataset=series_info['dataset'],
                protocol_name=series_info['protocol_name'],
                series_desc=series_info['series_desc']
            )
        
        # PERFORMANCE FIX: Minimal lock time - only for structure updates
        with self.progress_lock:
            # Fast assignments inside lock
            if needs_patient:
                new_study.series[series_uid] = new_series
                new_patient.studies[study_uid] = new_study
                collected_data[pat_id] = new_patient
            elif needs_study:
                new_study.series[series_uid] = new_series
                collected_data[pat_id].studies[study_uid] = new_study
            elif needs_series:
                collected_data[pat_id].studies[study_uid].series[series_uid] = new_series
            else:
                # Just append file to existing series
                collected_data[pat_id].studies[study_uid].series[series_uid].files.append(series_info['file_path'])
    
    @measure
    def _process_single_file(self, file_path: str) -> Optional[Dict]:
        """Process a single DICOM file - вынесено в отдельный метод для параллельности."""
        try:
            # Используем кэшированное чтение заголовка
            ds = _cached_dicom_header(file_path)
            if ds is None:
                return None
            
            # Извлекаем информацию о серии
            series_info = self._extract_series_info(ds, file_path)
            return series_info
            
        except Exception as e:
            logger.debug(f"Skipping file {file_path}: {e}")
            return None
    
    @measure 
    def _extract_series_info(self, ds: pydicom.Dataset, file_path: str) -> Optional[Dict]:
        """Extract series information from DICOM dataset."""
        pat_id = get_dicom_value(ds, (0x0010, 0x0020), "UNKNOWN_PATIENT_ID")
        study_uid = get_dicom_value(ds, (0x0020, 0x000D), "UNKNOWN_STUDY_UID")
        series_uid = get_dicom_value(ds, (0x0020, 0x000E), "UNKNOWN_SERIES_UID")
        
        if any(val.startswith("UNKNOWN_") for val in [pat_id, study_uid, series_uid]):
            logger.warning(f"Skipped file {file_path}: missing PatientID, StudyUID or SeriesUID.")
            return None
        
        study_date_str = get_dicom_value(ds, (0x0008, 0x0020), "00000000")
        study_time_raw = get_dicom_value(ds, (0x0008, 0x0030), "")
        
        # PERFORMANCE FIX: Robust time parsing with error suppression
        # Handle missing or malformed time without excessive logging
        if isinstance(study_time_raw, str) and study_time_raw:
            study_time_str = study_time_raw.split('.')[0]
            # Ensure time has at least 6 digits (HHMMSS)
            if len(study_time_str) < 6:
                study_time_str = study_time_str.ljust(6, '0')
        else:
            study_time_str = "000000"  # Default to midnight if time is missing
        
        # Ensure date is valid format
        if not isinstance(study_date_str, str) or len(study_date_str) < 8:
            study_date_str = "00000000"
        
        try:
            # Only try parsing if we have valid date format
            if study_date_str != "00000000":
                study_datetime = datetime.strptime(f"{study_date_str[:8]}{study_time_str[:6]}", "%Y%m%d%H%M%S")
            else:
                study_datetime = datetime.min
        except ValueError:
            # PERFORMANCE FIX: Use error suppression to reduce log spam
            error_key = f"invalid_datetime_{study_date_str}_{study_time_str[:6] if len(study_time_str) >= 6 else study_time_str}"
            log_parsing_error_once(error_key, study_uid, logger)
            study_datetime = datetime.min
        
        series_num_val = get_dicom_value(ds, (0x0020, 0x0011))
        try:
            series_number = int(series_num_val) if series_num_val is not None else float('inf')
        except ValueError:
            series_number = float('inf')
            logger.warning(f"SeriesNumber '{series_num_val}' for series {series_uid} is not a number.")
        
        return {
            'pat_id': pat_id,
            'study_uid': study_uid,
            'series_uid': series_uid,
            'study_datetime': study_datetime,
            'series_number': series_number,
            'dataset': ds,
            'protocol_name': normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), "")),
            'series_desc': normalize_dicom_text(get_dicom_value(ds, (0x0008, 0x103E), "")),
            'file_path': file_path 
        }
    
    @measure
    def _add_to_collected_data(self, collected_data: Dict, series_info: Dict, file_path: str):
        """Add series information to collected data structure."""
        pat_id = series_info['pat_id']
        study_uid = series_info['study_uid']
        series_uid = series_info['series_uid']
        
        # Initialize patient data if not exists
        if pat_id not in collected_data:
            collected_data[pat_id] = PatientData(original_id=pat_id, studies={})
        
        # Initialize study data if not exists
        if study_uid not in collected_data[pat_id].studies:
            collected_data[pat_id].studies[study_uid] = StudyInfo(
                uid=study_uid,
                series={},
                study_datetime=series_info['study_datetime']
            )
        
        # Initialize series data if not exists
        if series_uid not in collected_data[pat_id].studies[study_uid].series:
            collected_data[pat_id].studies[study_uid].series[series_uid] = SeriesInfo(
                uid=series_uid,
                files=[],
                series_number=series_info['series_number'],
                study_datetime=series_info['study_datetime'],
                first_dataset=series_info['dataset'],
                protocol_name=series_info['protocol_name'],
                series_desc=series_info['series_desc']
            )
        
        # Add file to series
        collected_data[pat_id].studies[study_uid].series[series_uid].files.append(file_path)


# --- Parallel Processing Functions ---
def process_patient_worker(args):
    """
    Обработка одного пациента в отдельном процессе.
    Функция верхнего уровня для использования с ProcessPoolExecutor.
    """
    patient_id, patient_files_info, bids_sub_id, output_dir, action_type = args

    original_patient_id = patient_id
    
    # Настраиваем логирование для процесса
    worker_logger = logging.getLogger(f"worker_{mp.current_process().pid}")
    worker_logger.setLevel(logging.INFO)

    # Логирование маппинга пациента
    logger = logging.getLogger(__name__)  # Используем основной логгер
    logger.info(f"Processing patient: {patient_id} -> {bids_sub_id}")
    
    try:
        # Создаем временный экземпляр органайзера для этого процесса
        temp_organizer = BidsOrganizer(output_dir, action_type)
        
        # Восстанавливаем структуру данных пациента из файлов
        patient_data = PatientData(original_id=patient_id, studies={})
        
        for study_uid, study_info in patient_files_info['studies'].items():
            study = StudyInfo(
                uid=study_uid,
                series={},
                study_datetime=datetime.fromisoformat(study_info['study_datetime'])
            )
            
            for series_uid, series_info in study_info['series'].items():
                # PERFORMANCE FIX: Use cache in worker process
                # Cannot pass Dataset between processes, so read from cache
                first_file = series_info['files'][0]
                cached_dataset = _cached_dicom_header(first_file)
                if cached_dataset is None:
                    continue
                
                series = SeriesInfo(
                    uid=series_uid,
                    files=series_info['files'],
                    series_number=series_info['series_number'],
                    study_datetime=study.study_datetime,
                    first_dataset=cached_dataset,
                    protocol_name=series_info['protocol_name'],
                    series_desc=series_info['series_desc']
                )
                study.series[series_uid] = series
            
            patient_data.studies[study_uid] = study
        
        # Обрабатываем пациента
        session_bids_map = temp_organizer._create_session_bids_mapping(patient_data.studies)

        # Логирование маппинга сессий для этого пациента
        logger.info(f"  Patient {bids_sub_id} has {len(session_bids_map)} sessions:")
        for study_uid, bids_ses_id in sorted(session_bids_map.items(), 
                                        key=lambda x: x[1]):  # Сортируем по ses ID
            study_info = patient_data.studies[study_uid]
            study_date = study_info.study_datetime.strftime("%Y-%m-%d")
            logger.info(f"    Session: {study_uid[:20]}... ({study_date}) -> {bids_ses_id}")

        # Создаем session_mapping для этого пациента
        session_mapping = {}
        for study_uid, bids_ses_id in session_bids_map.items():
            study_info = patient_data.studies[study_uid]
            study_date = study_info.study_datetime.strftime("%Y-%m-%d")
            
            # Создаем уникальный ключ для session mapping
            session_key = f"{patient_id}_{study_uid}"
            session_mapping[session_key] = {
                'original_patient_id': patient_id,
                'bids_patient_id': bids_sub_id,
                'original_study_uid': study_uid,
                'original_study_date': study_date,
                'bids_session_id': bids_ses_id
            }
        
        # Собираем результаты обработки
        results = {
            'patient_id': patient_id,
            'original_patient_id': original_patient_id,
            'bids_sub_id': bids_sub_id,
            'sessions_processed': 0,
            'missing_modalities': {},
            'failed_sessions': [],
            'errors': [],
            'selection_logs': [],
            'sessions_with_missing': [],
            'sessions_completely_missing': [],
            'session_mapping': session_mapping 
        }
        
        # Обрабатываем каждую сессию
        for study_info in patient_data.studies.values():
            try:
                has_missing, has_any = temp_organizer._process_study(
                    study_info, bids_sub_id, session_bids_map, patient_id
                )
                results['sessions_processed'] += 1
                
                if has_missing:
                    results['missing_modalities'][study_info.uid] = True
                if not has_any:
                    results['failed_sessions'].append(study_info.uid)

                # После каждого вызова _process_study проверяем failed_cases органайзера
                if temp_organizer.failed_cases['sessions_with_missing_modalities']:
                    # Берем последнюю добавленную запись (она относится к текущей сессии)
                    for session_info in temp_organizer.failed_cases['sessions_with_missing_modalities']:
                        if session_info['original_study_uid'] == study_info.uid:
                            results['sessions_with_missing'].append(session_info)
                            
                if temp_organizer.failed_cases['sessions_completely_missing']:
                    # Аналогично для полностью пропущенных сессий
                    for session_info in temp_organizer.failed_cases['sessions_completely_missing']:
                        if session_info['original_study_uid'] == study_info.uid:
                            results['sessions_completely_missing'].append(session_info)
                    
            except Exception as e:
                results['errors'].append(f"Error processing study {study_info.uid}: {str(e)}")
                worker_logger.error(f"Error processing study {study_info.uid}: {e}")
                import traceback
                worker_logger.error(f"Traceback: {traceback.format_exc()}")

        results['selection_logs'] = temp_organizer.selection_log

        # Collect detected modalities from selection logs
        detected_modalities = list(set([log.modality for log in temp_organizer.selection_log]))
        results['detected_modalities'] = detected_modalities
        results['num_series_organized'] = len(temp_organizer.selection_log)

        results['selection_logs'] = temp_organizer.selection_log

        # ДОБАВИТЬ: Итоговое логирование для пациента
        logger.info(f"  Completed patient {bids_sub_id}:")
        logger.info(f"    Sessions processed: {results['sessions_processed']}")
        if results['sessions_with_missing']:
            logger.info(f"    Sessions with missing modalities: {len(results['sessions_with_missing'])}")
        if results['errors']:
            logger.warning(f"    Errors encountered: {len(results['errors'])}")
        logger.info("-" * 50)  # Разделитель между пациентами
        
        return results
        
    except Exception as e:
        worker_logger.error(f"Error processing patient {patient_id}: {e}")
        return {
            'patient_id': patient_id,
            'bids_sub_id': bids_sub_id,
            'error': str(e)
        }

# --- Enhanced BIDS Organizer Class ---
class BidsOrganizer:
    """Enhanced BIDS organizer with proper naming conventions and missing modality handling."""
    
    def __init__(self, output_dir: str, action_type: str = 'copy', max_parallel_files: int = 20, max_workers: int = None, metrics_callback=None, streaming_mode: bool = False):
        self.output_dir = output_dir
        self.action_type = action_type
        self.detector = EnhancedModalityDetector()
        self.selection_log = []
        self.max_parallel_files = max_parallel_files  # Максимальное количество параллельных операций
        self.max_workers = max_workers  # Для параллельной обработки пациентов
        self.metrics_callback = metrics_callback  # Callback for research metrics
        self.streaming_mode = streaming_mode  # PERFORMANCE FIX: Enable streaming processing
        
        # BIDS modality mapping
        self.bids_modality_map = {
            't1': 'T1w',
            't1c': 'T1w-Gd',  # BIDS extension for post-contrast T1
            't2': 'T2w', 
            't2fl': 'FLAIR'
        }
        
        # Initialize mapping and failed cases tracking
        self.patient_mapping = {}
        self.session_mapping = {}
        self.failed_cases = {
            'patients_with_missing_modalities': {},
            'sessions_with_missing_modalities': [],
            'patients_completely_missing': [],
            'sessions_completely_missing': []
        }
        self.input_stats = {'total_patients': 0, 'total_sessions': 0}

    def _parallel_copy_files(self, file_operations: List[Tuple[str, str]], operation_type: str = 'copy'):
        """
        Параллельное копирование/перемещение файлов.
        
        Args:
            file_operations: Список кортежей (source_path, destination_path)
            operation_type: 'copy' или 'move'
        """
        # PERFORMANCE FIX: Use all available threads for file operations
        max_workers = min(self.max_parallel_files, 20)  # Use all 20 threads for file copying
        
        # Счетчики для логирования прогресса
        total_files = len(file_operations)
        completed = 0
        failed = 0
        lock = threading.Lock()
        
        def copy_single_file(src_dst_pair):
            """Копирование одного файла с обработкой ошибок."""
            src, dst = src_dst_pair
            try:
                # PERFORMANCE FIX: Optimize I/O operations for large datasets
                os.makedirs(os.path.dirname(dst), exist_ok=True)  # Ensure directory exists
                
                if operation_type == 'move':
                    shutil.move(src, dst)
                else:
                    # PERFORMANCE FIX: Use copy2 with buffering for better I/O performance
                    shutil.copy2(src, dst)  # copy2 preserves metadata and uses larger buffers
                
                # Обновляем счетчик
                nonlocal completed
                with lock:
                    completed += 1
                    if completed % 100 == 0:  # Логируем каждые 100 файлов
                        logger.debug(f"Progress: {completed}/{total_files} files processed")
                
                return True, src, dst
            except Exception as e:
                nonlocal failed
                with lock:
                    failed += 1
                return False, src, dst, str(e)
        
        # Выполняем параллельное копирование
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем все задачи
            futures = {executor.submit(copy_single_file, pair): pair 
                    for pair in file_operations}
            
            # Собираем результаты
            errors = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result[0]:  # Если была ошибка
                        _, src, dst, error = result
                        errors.append((src, dst, error))
                        logger.error(f"Failed to {operation_type} {src} -> {dst}: {error}")
                except Exception as e:
                    pair = futures[future]
                    errors.append((pair[0], pair[1], str(e)))
                    logger.error(f"Unexpected error processing {pair[0]}: {e}")
        
        # Финальное логирование
        logger.info(f"Parallel {operation_type} completed: {completed} successful, {failed} failed")
        
        return errors
    
    def _process_patients_parallel(self, patient_tasks: List[tuple], patient_bids_map: Dict[str, str], max_workers: int = None):
        """Параллельная обработка пациентов."""
        results_collected = []
        errors = []

        # PERFORMANCE FIX: Reduce concurrent I/O pressure to prevent disk saturation
        # Too many concurrent writes cause I/O bottleneck after 70 patients
        if max_workers is None:
            max_workers = min(8, mp.cpu_count())  # Limit to 8 concurrent patients to reduce I/O pressure
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем задачи
            future_to_patient = {
                executor.submit(process_patient_worker, task): task[0]  # patient_id
                for task in patient_tasks
            }
            
            # Собираем результаты по мере готовности
            completed_count = 0
            for future in as_completed(future_to_patient):
                patient_id = future_to_patient[future]
                completed_count += 1

                try:
                    result = future.result(timeout=300)  # 5 минут таймаут на пациента
                    results_collected.append(result)
                    
                    # Логируем прогресс
                    bids_id = patient_bids_map[patient_id]
                    logger.info(f"[{completed_count}/{len(patient_tasks)}] "
                          f"Completed processing {patient_id} ({bids_id})")
                    
                    # Обновляем статистику failed cases на основе результатов
                    if 'error' in result:
                        errors.append(f"Patient {patient_id}: {result['error']}")
                    else:
                        self._update_failed_cases_from_result(result)

                    if 'selection_logs' in result and result['selection_logs']:
                        self.selection_log.extend(result['selection_logs'])

                    # Call metrics callback if provided
                    if self.metrics_callback and 'error' not in result:
                        callback_data = {
                            'patient_id': patient_id,
                            'detected_modalities': result.get('detected_modalities', []),
                            'num_series_organized': result.get('num_series_organized', 0),
                            'sessions_processed': result.get('sessions_processed', 0),
                            'processing_status': 'success' if result.get('sessions_processed', 0) > 0 else 'failed',
                            'sessions_with_missing': result.get('sessions_with_missing', []),
                            'failure_reason': None if result.get('sessions_processed', 0) > 0 else 'No sessions processed'
                        }
                        self.metrics_callback(patient_id, callback_data)

                    # ДОБАВИТЬ: Собираем session_mapping из результата
                    if 'session_mapping' in result and result['session_mapping']:
                        self.session_mapping.update(result['session_mapping'])
                        
                except Exception as e:
                    logger.error(f"Failed to process patient {patient_id}: {e}")
                    errors.append(f"Patient {patient_id}: {str(e)}")
        
        if errors:
            logger.error(f"Errors during parallel processing: {len(errors)} patients failed")
            for error in errors[:5]:  # Показываем первые 5 ошибок
                logger.error(f"  - {error}")
    
    def _process_patients_streaming(self, patient_tasks: List[tuple], patient_bids_map: Dict[str, str]):
        """
        PERFORMANCE FIX: Sequential streaming processing to prevent memory buildup.
        Processes patients one by one with memory cleanup between patients.
        """
        logger.info("Using streaming mode - processing patients sequentially with memory cleanup")
        
        total_patients = len(patient_tasks)
        for idx, task in enumerate(patient_tasks, 1):
            patient_id = task[0]
            bids_id = patient_bids_map[patient_id]
            
            logger.info(f"[{idx}/{total_patients}] Processing patient {patient_id} ({bids_id}) in streaming mode")
            
            try:
                # Process single patient
                result = process_patient_worker(task)
                
                # Handle result
                if 'error' in result:
                    logger.error(f"Patient {patient_id}: {result['error']}")
                else:
                    self._update_failed_cases_from_result(result)
                    
                    if 'selection_logs' in result and result['selection_logs']:
                        self.selection_log.extend(result['selection_logs'])
                    
                    # Call metrics callback if provided
                    if self.metrics_callback:
                        callback_data = {
                            'patient_id': patient_id,
                            'detected_modalities': result.get('detected_modalities', []),
                            'num_series_organized': result.get('num_series_organized', 0),
                            'sessions_processed': result.get('sessions_processed', 0),
                            'processing_status': 'success' if result.get('sessions_processed', 0) > 0 else 'failed',
                            'sessions_with_missing': result.get('sessions_with_missing', []),
                            'failure_reason': None if result.get('sessions_processed', 0) > 0 else 'No sessions processed'
                        }
                        self.metrics_callback(patient_id, callback_data)
                    
                    # Collect session mapping
                    if 'session_mapping' in result and result['session_mapping']:
                        self.session_mapping.update(result['session_mapping'])
                
                logger.info(f"  Completed patient {patient_id} ({bids_id})")
                
                # PERFORMANCE FIX: Add I/O sync and reduce concurrent pressure periodically
                if idx % 25 == 0:  # Every 25 patients - balance between memory and I/O
                    logger.info(f"  Performing maintenance after {idx} patients...")
                    
                    # Force filesystem sync to prevent I/O queue buildup
                    try:
                        os.sync()  # Force kernel to flush dirty pages
                        logger.info(f"    Filesystem sync completed")
                    except:
                        pass  # sync() may not be available on all systems
                    
                    # Clear cache and garbage collect
                    clear_dicom_cache()
                    import gc
                    gc.collect()
                    logger.info(f"    Memory cleanup completed")
                    
                    # Brief pause to let I/O subsystem recover
                    import time
                    time.sleep(1)
                    logger.info(f"  Maintenance completed, resuming processing...")
                    
            except Exception as e:
                logger.error(f"Failed to process patient {patient_id} in streaming mode: {e}")
        
        logger.info("Streaming processing completed")
    
    @measure(capture_args=True) 
    def organize_to_bids(self, collected_data: Dict[str, PatientData]):
        """Organize collected data into BIDS structure with enhanced logging."""
        logger.info("Phase 2: Creating BIDS structure and organizing files...")

        # Начало фазы 2
        profiler.record_phase("phase_2_organizing", "start")
        profiler.memory_checkpoint("organize_start")
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot create output directory {self.output_dir}: {e}")
            raise
        
        # Create logs directory
        logs_dir = os.path.join(self.output_dir, 'logs')
        try:
            os.makedirs(logs_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot create logs directory {logs_dir}: {e}")
            raise
        
        # Store input statistics for comparison
        self.input_stats = {
            'total_patients': len(collected_data),
            'total_sessions': sum(len(patient.studies) for patient in collected_data.values())
        }
        
        # Create BIDS patient IDs
        patient_bids_map = self._create_patient_bids_mapping(collected_data)

        # ДОБАВИТЬ: Логирование общего маппинга пациентов
        logger.info("=" * 50)
        logger.info("PATIENT MAPPING:")
        logger.info("=" * 50)
        for orig_id, bids_id in sorted(patient_bids_map.items(), 
                                    key=lambda x: x[1]):  # Сортируем по BIDS ID
            patient_data = collected_data[orig_id]
            num_sessions = len(patient_data.studies)
            logger.info(f"  {orig_id} -> {bids_id} ({num_sessions} sessions)")
        logger.info("=" * 50)
        
        #  ИЗМЕНЕНИЕ: Подготовка данных для параллельной обработки
        # Конвертируем данные в сериализуемый формат
        patient_tasks = []
        
        for patient_id, patient_data in collected_data.items():
            bids_sub_id = patient_bids_map[patient_id]
            
            # Преобразуем в сериализуемый формат (только пути и метаданные)
            patient_files_info = {
                'studies': {}
            }
            
            for study_uid, study_info in patient_data.studies.items():
                patient_files_info['studies'][study_uid] = {
                    'study_datetime': study_info.study_datetime.isoformat(),
                    'series': {}
                }
                
                for series_uid, series_info in study_info.series.items():
                    patient_files_info['studies'][study_uid]['series'][series_uid] = {
                        'files': series_info.files,
                        'series_number': series_info.series_number,
                        'protocol_name': series_info.protocol_name,
                        'series_desc': series_info.series_desc
                        # NOTE: Cannot pass pydicom.Dataset between processes - not serializable
                        # Will use cache in worker process instead
                    }
            
            patient_tasks.append((
                patient_id,
                patient_files_info,
                bids_sub_id,
                self.output_dir,
                self.action_type
            ))
        
        # Всегда используем параллельную обработку
        if self.max_workers:
            max_workers = self.max_workers
        else:
            max_workers = min(20, len(patient_tasks))  # Use all 20 cores up to number of patients

        # Но если только 1 пациент, используем 1 воркер
        if len(patient_tasks) == 1:
            max_workers = 1
            
        logger.info(f"Processing {len(patient_tasks)} patients using {max_workers} workers")

        with profiler.measure_block("patient_processing"):
            # PERFORMANCE FIX: ALWAYS use parallel processing - streaming is too slow!
            # Streaming mode causes 10-minute delays per patient after cache clear
            logger.info(f"Using parallel processing for maximum performance ({max_workers} workers)")
            self._process_patients_parallel(patient_tasks, patient_bids_map, self.max_workers or max_workers)
        
        # Generate selection summary
        self._generate_selection_summary()
        
        # Write mapping and failed cases files
        self._write_mapping_files(logs_dir)

        # Конец фазы 2
        profiler.record_phase("phase_2_organizing", "end")
        profiler.memory_checkpoint("organize_end")
        
        logger.info("Enhanced BIDS organization completed!")
    
    @measure
    def _create_patient_bids_mapping(self, collected_data: Dict) -> Dict[str, str]:
        """Create mapping from original patient IDs to BIDS IDs."""
        sorted_patient_ids = sorted(collected_data.keys())
        mapping = {orig_id: f"sub-{i+1:03d}" for i, orig_id in enumerate(sorted_patient_ids)}
        
        # Store in instance variable for later export
        self.patient_mapping = mapping
        
        return mapping
    
    def _create_session_bids_mapping(self, studies: Dict[str, StudyInfo]) -> Dict[str, str]:
        """Create mapping from original study UIDs to BIDS session IDs."""
        sorted_study_uids = sorted(studies.keys(), key=lambda uid: studies[uid].study_datetime)
        return {orig_id: f"ses-{i+1:03d}" for i, orig_id in enumerate(sorted_study_uids)}
    
    @measure(capture_args=True)
    def _process_study(self, study_info: StudyInfo, bids_sub_id: str, session_bids_map: Dict[str, str], 
                      original_patient_id: str) -> Tuple[bool, bool]:
        """Process a single study/session with enhanced modality detection.
        Returns (has_missing_modalities, has_any_modalities)."""
        bids_ses_id = session_bids_map[study_info.uid]
        session_id = f"{bids_sub_id}_{bids_ses_id}"
        
        study_date = study_info.study_datetime.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"  Processing session {bids_ses_id} for {bids_sub_id}:")
        logger.info(f"    Original Study UID: {study_info.uid[:30]}...")
        logger.info(f"    Study Date/Time: {study_date}")
        logger.info(f"    Number of series: {len(study_info.series)}")
        
        # Group series by modality with enhanced detection
        modality_groups, detection_details = self._group_series_by_modality_enhanced(study_info.series, session_id)
        
        # Required modalities
        required_modalities = ['t1', 't1c', 't2', 't2fl']
        found_modalities = list(modality_groups.keys())
        missing_modalities = [m for m in required_modalities if m not in found_modalities]
        
        logger.info(f"    Found modalities: {', '.join(found_modalities) if found_modalities else 'None'}")
        
        has_missing_modalities = False
        has_any_modalities = len(found_modalities) > 0
        
        if missing_modalities:
            has_missing_modalities = True
            logger.warning(f"    Missing modalities: {', '.join(missing_modalities)}")
            
            # Record failed session
            study_date = study_info.study_datetime.strftime("%Y-%m-%d")
            session_failure_info = {
                'original_patient_id': original_patient_id,
                'bids_patient_id': bids_sub_id,
                'original_study_uid': study_info.uid,
                'original_study_date': study_date,
                'bids_session_id': bids_ses_id,
                'found_modalities': found_modalities,
                'missing_modalities': missing_modalities
            }
            self.failed_cases['sessions_with_missing_modalities'].append(session_failure_info)
            
            # Update patient-level tracking
            if original_patient_id in self.failed_cases['patients_with_missing_modalities']:
                self.failed_cases['patients_with_missing_modalities'][original_patient_id]['sessions_with_issues'].append({
                    'session_id': bids_ses_id,
                    'study_date': study_date,
                    'missing_modalities': missing_modalities
                })
        
        # Track completely missing sessions (no modalities at all)
        if not has_any_modalities:
            study_date = study_info.study_datetime.strftime("%Y-%m-%d")
            self.failed_cases['sessions_completely_missing'].append({
                'original_patient_id': original_patient_id,
                'bids_patient_id': bids_sub_id,
                'original_study_uid': study_info.uid,
                'original_study_date': study_date,
                'bids_session_id': bids_ses_id,
                'total_series': len(study_info.series),
                'reason': 'No valid modalities detected'
            })
        
        # Only create directories and process found modalities
        if found_modalities:
            bids_anat_path = os.path.join(self.output_dir, bids_sub_id, bids_ses_id, 'anat')
            
            # Process each found modality
            for modality in found_modalities:
                series_with_strategies = modality_groups[modality]
                self._process_modality_group_enhanced(series_with_strategies, modality, bids_anat_path, 
                                                   bids_sub_id, bids_ses_id, session_id)
        else:
            logger.warning(f"    No valid modalities found for session {bids_ses_id} - skipping directory creation")
        
        # Log session summary
        logger.info(f"    Session {bids_ses_id} processing complete. "
                   f"Processed {len(found_modalities)}/{len(required_modalities)} modalities.")
        
        return has_missing_modalities, has_any_modalities
    
    @measure
    def _group_series_by_modality_enhanced(self, series_dict: Dict[str, SeriesInfo], session_id: str) -> Tuple[Dict, Dict]:
        """Enhanced grouping with detailed logging."""
        modality_groups = defaultdict(list)
        detection_details = {}

        # Логирование начала определения модальностей
        logger.info(f"    Determining modalities for {len(series_dict)} series:")
        
        for series_info in series_dict.values():
            modality, strategy, details = self.detector.determine_modality_with_details(
                series_info.first_dataset, series_info.files[0], session_id
            )
            
            detection_details[series_info.uid] = {
                'modality': modality,
                'strategy': strategy.get_name() if strategy else None,
                'details': details
            }
            
            if modality == 'unknown':
                logger.warning(f"    Skipping series {series_info.uid}: unknown modality.")
                logger.debug(f"      Protocol: '{series_info.protocol_name}'")
                logger.debug(f"      Series Desc: '{series_info.series_desc}'")
                continue
            
            modality_groups[modality].append((series_info, strategy))
            logger.debug(f"    Series {series_info.uid} -> {modality} (strategy: {strategy.get_name() if strategy else 'None'})")

        # Сводка по найденным модальностям
        if modality_groups:
            found_modalities = list(modality_groups.keys())
            logger.info(f"    Found modalities: {', '.join(sorted(found_modalities))}")
        else:
            logger.warning(f"    No valid modalities found")
        
        return dict(modality_groups), detection_details
    
    @measure
    def _process_modality_group_enhanced(self, series_with_strategies: List[Tuple[SeriesInfo, ModalityDetectionStrategy]], 
                                       modality: str, bids_anat_path: str, bids_sub_id: str, bids_ses_id: str, session_id: str):
        """Enhanced modality group processing with priority scoring."""
        logger.debug(f"    Processing {len(series_with_strategies)} series for modality: {modality}")
        
        if len(series_with_strategies) == 1:
            # Only one series - use it directly
            series_info, strategy = series_with_strategies[0]
            selected_series = [series_info]
            selection_reason = "only_candidate"
            logger.info(f"    Selected only series for {modality}: {series_info.uid}")
        else:
            # Multiple series - apply enhanced selection
            selected_series, selection_reason, scoring_details = self._apply_enhanced_priority_selection(
                series_with_strategies, modality, session_id
            )
        
        # Create modality directory
        bids_modality_dir = os.path.join(bids_anat_path, modality)
        try:
            os.makedirs(bids_modality_dir, exist_ok=True)
            logger.debug(f"    Created directory: {bids_modality_dir}")
        except OSError as e:
            logger.error(f"    Cannot create directory {bids_modality_dir}: {e}")
            return
        
        # Process selected series
        for series_info in selected_series:
            # Log selection decision
            # Find the strategy used for this series
            strategy_used = None
            for series, strategy in series_with_strategies:
                if series.uid == series_info.uid:
                    strategy_used = strategy.get_name() if strategy else "Unknown"
                    break
            
            selection_log_entry = ModalitySelectionLog(
                session_id=session_id,
                modality=modality,
                selected_protocol=series_info.protocol_name,
                selection_reason=selection_reason,
                strategy_used=strategy_used or "Unknown",
                candidates_considered=[series.protocol_name for series, _ in series_with_strategies],
                forbidden_filtered=[],  # Would need to track this in detection
                priority_scores=getattr(self, '_last_scoring_details', {}),
                year_detected=None  # Would extract from study date
            )
            self.selection_log.append(selection_log_entry)
            
            self._copy_series_files_enhanced(series_info, modality, bids_modality_dir, 
                                           bids_sub_id, bids_ses_id)
    
    @measure
    def _apply_enhanced_priority_selection(self, series_with_strategies: List[Tuple[SeriesInfo, ModalityDetectionStrategy]], 
                                         modality: str, session_id: str) -> Tuple[List[SeriesInfo], str, Dict]:
        """Enhanced priority selection with scoring system."""
        logger.debug(f"    Found {len(series_with_strategies)} series for {modality}, applying enhanced selection")
        
        # Extract series and strategies
        series_list = [series for series, _ in series_with_strategies]
        strategies = [strategy for _, strategy in series_with_strategies if strategy]
        
        # Collect scoring weights from all applicable strategies
        all_scoring_weights = {}
        for strategy in strategies:
            weights = strategy.get_scoring_weights(modality)
            all_scoring_weights.update(weights)
        
        # Calculate scores for each series
        series_scores = {}
        scoring_details = {}
        
        for series_info in series_list:
            score = self._calculate_series_score(series_info, modality, all_scoring_weights)
            series_scores[series_info.uid] = score
            scoring_details[series_info.uid] = {
                'protocol': series_info.protocol_name,
                'series_desc': series_info.series_desc,
                'score': score
            }
            logger.debug(f"      Series {series_info.uid}: score={score:.2f} (Protocol: '{series_info.protocol_name}')")
        
        # Store for logging
        self._last_scoring_details = scoring_details
        
        # Select highest scoring series
        if series_scores:
            best_series_uid = max(series_scores.keys(), key=lambda uid: series_scores[uid])
            best_series = next(s for s in series_list if s.uid == best_series_uid)
            
            logger.info(f"    Selected series by scoring for {modality}: {best_series_uid} "
                       f"(score: {series_scores[best_series_uid]:.2f})")
            return [best_series], "priority_scoring", scoring_details
        
        # Fallback to newest if scoring fails
        logger.debug(f"    Falling back to newest acquisition for {modality}")
        newest_series = self._select_newest_series(series_list)
        return [newest_series], "newest_acquisition", {}
    
    def _calculate_series_score(self, series_info: SeriesInfo, modality: str, scoring_weights: Dict[str, float]) -> float:
        """Calculate priority score for a series."""
        score = 1.0  # Base score
        
        # Combine protocol name and series description for analysis
        combined_text = f"{series_info.protocol_name} {series_info.series_desc}".lower()
        
        # Apply scoring weights
        for keyword, weight in scoring_weights.items():
            if keyword in combined_text:
                score *= weight
                logger.debug(f"        Applied weight {weight} for keyword '{keyword}'")
        
        # Bonus for brain-specific sequences
        if 'brain' in combined_text:
            score *= 1.1
            logger.debug(f"        Applied brain bonus: 1.1")
        
        # Penalty for spine/other anatomy
        anatomy_penalties = {'spine': 0.8, 'cervical': 0.8, 'pit': 0.7, 'pituitary': 0.7}
        for anatomy, penalty in anatomy_penalties.items():
            if anatomy in combined_text:
                score *= penalty
                logger.debug(f"        Applied {anatomy} penalty: {penalty}")
        
        return score
    
    def _select_newest_series(self, series_list: List[SeriesInfo]) -> SeriesInfo:
        """Select the newest series by acquisition time."""
        series_with_times = []
        
        for series in series_list:
            # Get acquisition datetime from the first dataset
            ds = series.first_dataset
            acq_date = get_dicom_value(ds, (0x0008, 0x0022), "")  # AcquisitionDate
            acq_time = get_dicom_value(ds, (0x0008, 0x0032), "").split('.')[0]  # AcquisitionTime
            
            # Fallback hierarchy for datetime
            if not acq_date:
                acq_date = get_dicom_value(ds, (0x0008, 0x0021), "")  # SeriesDate
                acq_time = get_dicom_value(ds, (0x0008, 0x0031), "").split('.')[0]  # SeriesTime
            
            if not acq_date:
                acq_date = get_dicom_value(ds, (0x0008, 0x0020), "00000000")  # StudyDate
                acq_time = get_dicom_value(ds, (0x0008, 0x0030), "000000").split('.')[0]  # StudyTime
            
            # Parse datetime
            try:
                if len(acq_date) >= 8 and len(acq_time) >= 6:
                    acq_datetime = datetime.strptime(f"{acq_date[:8]}{acq_time[:6]}", "%Y%m%d%H%M%S")
                else:
                    acq_datetime = series.study_datetime
            except ValueError:
                acq_datetime = series.study_datetime
            
            series_with_times.append((series, acq_datetime))
        
        # Sort by acquisition time (newest first) and select the first
        series_with_times.sort(key=lambda x: x[1], reverse=True)
        selected_series = series_with_times[0][0]
        
        logger.info(f"    Selected newest series: {selected_series.uid} "
                   f"(acquired at {series_with_times[0][1]})")
        
        return selected_series
    
    @measure
    def _copy_series_files_enhanced(self, series_info: SeriesInfo, modality: str, bids_modality_dir: str,
                                bids_sub_id: str, bids_ses_id: str):
        """Copy files with proper BIDS naming conventions using parallel processing."""
        sorted_files = self._sort_files_by_instance_number(series_info.files)
        
        # BIDS suffix mapping
        bids_suffix = self.bids_modality_map.get(modality, modality)
        
        logger.info(f"    Copying {len(sorted_files)} files for {modality} -> {bids_suffix} "
                f"(Series UID: {series_info.uid})")
        
        # Подготавливаем список операций копирования
        file_operations = []
        for slice_idx, src_file_path in enumerate(sorted_files, 1):
            bids_filename = f"{bids_sub_id}_{bids_ses_id}_{bids_suffix}_instance-{slice_idx:03d}.dcm"
            dst_file_path = os.path.join(bids_modality_dir, bids_filename)
            file_operations.append((src_file_path, dst_file_path))
        
        # PERFORMANCE FIX: Always use parallel copying for ANY number of files
        # The 45-second bottleneck is here - sequential file copying!
        with profiler.measure_block(f"parallel_{self.action_type}_{len(sorted_files)}_files"):
            # ALWAYS use parallel copying - even for small series
            if len(file_operations) > 1:  # Use parallel for 2+ files
                errors = self._parallel_copy_files(file_operations, self.action_type)
                if errors:
                    logger.warning(f"    {len(errors)} files failed during {self.action_type}")
            else:
                # Only single file - do direct copy
                src_file_path, dst_file_path = file_operations[0]
                try:
                    # PERFORMANCE FIX: Ensure directory exists and optimize I/O
                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                    
                    if self.action_type == 'move':
                        shutil.move(src_file_path, dst_file_path)
                    else:
                        # PERFORMANCE FIX: Use copy2 for better I/O performance
                        shutil.copy2(src_file_path, dst_file_path)
                except Exception as e:
                    logger.error(f"Failed to {self.action_type} {src_file_path} -> {dst_file_path}: {e}")
    
    @measure
    def _sort_files_by_instance_number(self, dicom_files_paths: List[str]) -> List[str]:
        """Sort DICOM files by InstanceNumber."""
        sorted_files = []
        for f_path in dicom_files_paths:
            try:
                # PERFORMANCE FIX: Use cached header instead of re-reading file
                ds_slice = _cached_dicom_header(f_path)
                if ds_slice is None:
                    # Fallback to direct read if cache fails
                    ds_slice = pydicom.dcmread(f_path, stop_before_pixels=True, specific_tags=[(0x0020,0x0013)])
                instance_number = get_dicom_value(ds_slice, (0x0020,0x0013))
                if instance_number is not None:
                    try:
                        instance_number = int(instance_number)
                    except ValueError:
                        logger.warning(f"Cannot convert InstanceNumber '{instance_number}' to int for {f_path}. Using filename for sorting.")
                        instance_number = f_path
                else:
                    logger.warning(f"InstanceNumber missing in {f_path}. Using filename for sorting.")
                    instance_number = f_path
                sorted_files.append((instance_number, f_path))
            except Exception as e:
                logger.error(f"Error reading InstanceNumber from {f_path}: {e}. File will be sorted at the end.")
                sorted_files.append((float('inf'), f_path))

        # Sort: first by numeric InstanceNumber, then by file path
        sorted_files.sort(key=lambda x: (isinstance(x[0], str), x[0]))
        return [f_path for _, f_path in sorted_files]
    
    def _generate_selection_summary(self):
        """Generate and log selection summary."""
        if not self.selection_log:
            logger.warning("No modality selection logs to save")
            return
        
        logger.info("="*50)
        logger.info("MODALITY SELECTION SUMMARY")
        logger.info("="*50)
        
        # Group by session
        sessions = defaultdict(list)
        for log_entry in self.selection_log:
            sessions[log_entry.session_id].append(log_entry)
        
        for session_id, entries in sessions.items():
            logger.info(f"\nSession: {session_id}")
            for entry in entries:
                logger.info(f"  {entry.modality}: {entry.selected_protocol}")
                logger.info(f"    Strategy: {entry.strategy_used}")
                logger.info(f"    Reason: {entry.selection_reason}")
                if len(entry.candidates_considered) > 1:
                    logger.info(f"    Candidates: {len(entry.candidates_considered)} total")
        
        # Export detailed log to JSON
        logs_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)  # Убедимся что папка существует
        log_file = os.path.join(logs_dir, 'modality_selection_log.json')

        try:
            with open(log_file, 'w') as f:
                json.dump([{
                    'session_id': log.session_id,
                    'modality': log.modality,
                    'selected_protocol': log.selected_protocol,
                    'selection_reason': log.selection_reason,
                    'strategy_used': log.strategy_used,
                    'candidates_considered': log.candidates_considered,
                    'priority_scores': log.priority_scores
                } for log in self.selection_log], f, indent=2)
            logger.info(f"\nDetailed selection log saved to: {log_file}")
        except Exception as e:
            logger.error(f"Failed to save selection log: {e}")
    
    def _write_mapping_files(self, logs_dir: str):
        """Write mapping and failed cases files to logs directory."""
        # PERFORMANCE FIX: Log cache statistics and performance metrics
        cache_stats = get_cache_statistics()
        logger.info("=" * 50)
        logger.info("PERFORMANCE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"DICOM Cache Performance:")
        logger.info(f"  Cache hits: {cache_stats['dicom_cache_hits']}")
        logger.info(f"  Cache misses: {cache_stats['dicom_cache_misses']}")
        logger.info(f"  Cache size: {cache_stats['dicom_cache_size']} entries")
        if cache_stats['dicom_cache_hits'] + cache_stats['dicom_cache_misses'] > 0:
            hit_rate = cache_stats['dicom_cache_hits'] / (cache_stats['dicom_cache_hits'] + cache_stats['dicom_cache_misses']) * 100
            logger.info(f"  Hit rate: {hit_rate:.1f}%")
        
        logger.info(f"Patient Scan Cache:")
        logger.info(f"  Cached results: {cache_stats['patient_cache_size']} datasets")
        
        logger.info(f"Error Suppression:")
        logger.info(f"  Total errors suppressed: {cache_stats['total_errors_suppressed']}")
        logger.info(f"  Error types encountered: {cache_stats['error_types']}")
        logger.info("=" * 50)
        
        # Write patient and session mapping file
        mapping_file = os.path.join(logs_dir, 'bids_mapping.json')
        mapping_data = {
            'patients': self.patient_mapping,
            'sessions': self.session_mapping,
            'performance_stats': cache_stats  # Include performance data
        }
        
        try:
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            logger.info(f"BIDS mapping saved to: {mapping_file}")
            logger.info(f"  - Mapped {len(self.patient_mapping)} patients")
            logger.info(f"  - Mapped {len(self.session_mapping)} sessions")
        except Exception as e:
            logger.error(f"Failed to save BIDS mapping: {e}")
        
        # Write failed cases file
        failed_file = os.path.join(logs_dir, 'failed_cases.json')
        
        # Add summary statistics
        self.failed_cases['summary'] = {
            'input_patients': self.input_stats['total_patients'],
            'input_sessions': self.input_stats['total_sessions'],
            'output_patients': self.input_stats['total_patients'] - len(self.failed_cases['patients_completely_missing']),
            'output_sessions': self.input_stats['total_sessions'] - len(self.failed_cases['sessions_completely_missing']),
            'total_patients_with_partial_issues': len(self.failed_cases['patients_with_missing_modalities']),
            'total_sessions_with_partial_issues': len(self.failed_cases['sessions_with_missing_modalities']),
            'total_patients_completely_missing': len(self.failed_cases['patients_completely_missing']),
            'total_sessions_completely_missing': len(self.failed_cases['sessions_completely_missing'])
        }
        
        try:
            with open(failed_file, 'w') as f:
                json.dump(self.failed_cases, f, indent=2)
            logger.info(f"Failed cases report saved to: {failed_file}")
            
            # Log summary of failed cases
            logger.info(f"Input/Output Comparison:")
            logger.info(f"  Patients: {self.failed_cases['summary']['input_patients']} input -> {self.failed_cases['summary']['output_patients']} output")
            logger.info(f"  Sessions: {self.failed_cases['summary']['input_sessions']} input -> {self.failed_cases['summary']['output_sessions']} output")
            
            if self.failed_cases['summary']['total_patients_with_partial_issues'] > 0:
                logger.warning(f"Found {self.failed_cases['summary']['total_patients_with_partial_issues']} patients with partial missing modalities")
            
            if self.failed_cases['summary']['total_sessions_with_partial_issues'] > 0:
                logger.warning(f"Found {self.failed_cases['summary']['total_sessions_with_partial_issues']} sessions with partial missing modalities")
            
            if self.failed_cases['summary']['total_patients_completely_missing'] > 0:
                logger.warning(f"Found {self.failed_cases['summary']['total_patients_completely_missing']} patients completely missing from output")
            
            if self.failed_cases['summary']['total_sessions_completely_missing'] > 0:
                logger.warning(f"Found {self.failed_cases['summary']['total_sessions_completely_missing']} sessions completely missing from output")
        except Exception as e:
            logger.error(f"Failed to save failed cases report: {e}")

    def _update_failed_cases_from_result(self, result: Dict):
        """Обновляет статистику failed cases из результата обработки пациента."""
        patient_id = result['patient_id']
        bids_sub_id = result['bids_sub_id']
        
        # Обновляем статистику пропущенных модальностей для пациентов
        if result.get('missing_modalities'):
            if patient_id not in self.failed_cases['patients_with_missing_modalities']:
                self.failed_cases['patients_with_missing_modalities'][patient_id] = {
                    'bids_id': bids_sub_id,
                    'sessions_with_issues': []
                }
            
            # ДОБАВИТЬ: Заполняем sessions_with_issues из детальной информации
            if 'sessions_with_missing' in result:
                for session_info in result['sessions_with_missing']:
                    self.failed_cases['patients_with_missing_modalities'][patient_id]['sessions_with_issues'].append({
                        'session_id': session_info['bids_session_id'],
                        'study_date': session_info['original_study_date'],
                        'missing_modalities': session_info['missing_modalities']
                    })
        
        # ДОБАВИТЬ: Обновляем детальную информацию о сессиях с пропущенными модальностями
        if 'sessions_with_missing' in result:
            self.failed_cases['sessions_with_missing_modalities'].extend(result['sessions_with_missing'])
        
        # ДОБАВИТЬ: Обновляем полностью пропущенные сессии из результата
        if 'sessions_completely_missing' in result:
            self.failed_cases['sessions_completely_missing'].extend(result['sessions_completely_missing'])
        
        # Обновляем полностью пропущенные пациенты
        if 'failed_sessions' in result and len(result['failed_sessions']) == result['sessions_processed']:
            # Если все сессии пациента failed - пациент полностью пропущен
            self.failed_cases['patients_completely_missing'].append({
                'original_patient_id': patient_id,
                'bids_patient_id': bids_sub_id,
                'total_sessions': result['sessions_processed'],
                'reason': 'No sessions with valid modalities'
            })


# --- CLI Interface ---
def main():
    """Enhanced main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced DICOM to BIDS converter with intelligent modality detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dicom/input /path/to/bids/output
  %(prog)s /path/to/dicom/input /path/to/bids/output --action move --log-file conversion.log
  %(prog)s /path/to/dicom/input /path/to/bids/output --verbose
  %(prog)s /path/to/dicom/input /path/to/bids/output --dry-run
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing DICOM files'
    )
    
    parser.add_argument(
        'output_dir', 
        type=str,
        help='Output directory for BIDS structure'
    )
    
    # Optional arguments
    parser.add_argument(
        '--action',
        choices=['copy', 'move'],
        default='copy',
        help='Action to perform on files: copy (default) or move'
    )
    
    parser.add_argument(
        '--log_file',
        type=str,
        help='Path to log file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging to console'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without copying/moving files'
    )
    
    parser.add_argument(
        '--list-strategies',
        action='store_true',
        help='List all available detection strategies and exit'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Enhanced DICOM to BIDS Converter v3.0.0 (CSV-Analysis Based)'
    )

    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    parser.add_argument(
        '--profile-output-dir',
        type=str,
        default='profiling_reports',
        help='Directory for profiling reports (default: profiling_reports)'
    )
    
    parser.add_argument(
        '--profile-slow-threshold',
        type=float,
        default=1.0,
        help='Log operations slower than this threshold in seconds (default: 1.0)'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: number of CPU cores)'
    )
    
    parser.add_argument(
        '--streaming-mode',
        action='store_true',
        help='Enable streaming mode for large datasets (processes patients sequentially with memory cleanup)'
    )
    
    args = parser.parse_args()

    start_time = time.perf_counter()
    
    # Setup logging
    try:
        setup_logging(args.log_file)
        if args.verbose:
            console_handler = None
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    console_handler = handler
                    break
            if console_handler:
                console_handler.setLevel(logging.ERROR)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1)

    # Настройка профилирования
    if args.profile:
        setup_profiling(
            enabled=True,
            output_dir=args.profile_output_dir,
            log_slow_operations=args.profile_slow_threshold
        )
        # Начинаем сессию профилирования
        session_name = f"dicom_reorg_{os.path.basename(args.input_dir)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        profiler.start_session(session_name)
        logger.info(f"Performance profiling enabled. Reports will be saved to: {args.profile_output_dir}")
    
    # Handle --list-strategies
    if args.list_strategies:
        detector = EnhancedModalityDetector()
        print("\nEnhanced Detection Strategies (CSV-Analysis Based):")
        print("-" * 60)
        for strategy in detector.strategies:
            print(f"- {strategy.get_name()} (priority: {strategy.get_priority()})")
            if hasattr(strategy, 'target_years'):
                print(f"  Target years: {strategy.target_years}")
        print("\nStrategies are tried in order of priority (lower number = higher priority)")
        print("Year-specific strategies are exclusive - they override standard detection")
        sys.exit(0)
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory.")
        sys.exit(1)
    
    # Validate output directory parent exists
    output_parent = os.path.dirname(os.path.abspath(args.output_dir))
    if not os.path.exists(output_parent):
        print(f"Error: Parent directory of output '{output_parent}' does not exist.")
        sys.exit(1)
    
    # Log startup information
    logger.info("="*70)
    logger.info("Enhanced DICOM to BIDS Converter Started (v3.0.0 - CSV Analysis Based)")
    logger.info("="*70)
    logger.info(f"Input directory: {os.path.abspath(args.input_dir)}")
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Log file: {os.path.abspath(args.log_file)}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # List active strategies
    detector = EnhancedModalityDetector()
    logger.info("Active detection strategies:")
    for strategy in detector.strategies:
        logger.info(f"  - {strategy.get_name()} (priority: {strategy.get_priority()})")
    
    try:
        # Phase 1: Scan DICOM files
        # Общий замер времени выполнения
        with profiler.measure_block("total_execution"):
            scanner = DicomScanner()
            collected_data = scanner.scan_directory(args.input_dir)
            
            if not collected_data:
                logger.warning("No valid DICOM files found in input directory.")
                print("Warning: No valid DICOM files found.")
                sys.exit(0)
            
            # Log summary statistics
            total_patients = len(collected_data)
            total_studies = sum(len(patient.studies) for patient in collected_data.values())
            total_series = sum(
                len(study.series) 
                for patient in collected_data.values() 
                for study in patient.studies.values()
            )
            total_files = sum(
                len(series.files)
                for patient in collected_data.values()
                for study in patient.studies.values()
                for series in study.series.values()
            )
            
            logger.info(f"Scan Summary:")
            logger.info(f"  Patients: {total_patients}")
            logger.info(f"  Studies: {total_studies}")
            logger.info(f"  Series: {total_series}")
            logger.info(f"  Files: {total_files}")
            
            if args.dry_run:
                logger.info("DRY RUN MODE - No files will be copied or moved")
                
                # Show what would be processed
                temp_organizer = BidsOrganizer(args.output_dir, args.action)
                required_modalities = ['t1', 't1c', 't2', 't2fl']
                
                for patient_id, patient_data in collected_data.items():
                    logger.info(f"Patient: {patient_id}")
                    
                    for study_uid, study_info in patient_data.studies.items():
                        session_id = f"patient_{patient_id}_study_{study_uid[:8]}"
                        logger.info(f"  Study: {study_uid} ({study_info.study_datetime})")
                        
                        # Analyze what would be detected
                        modality_groups, detection_details = temp_organizer._group_series_by_modality_enhanced(
                            study_info.series, session_id
                        )
                        
                        found_modalities = list(modality_groups.keys())
                        missing_modalities = [m for m in required_modalities if m not in found_modalities]
                        
                        logger.info(f"    Would find: {', '.join(found_modalities) if found_modalities else 'None'}")
                        if missing_modalities:
                            logger.info(f"    Would be missing: {', '.join(missing_modalities)}")
                        
                        # Show selection details for found modalities
                        for modality in found_modalities:
                            series_list = [s for s, _ in modality_groups[modality]]
                            if len(series_list) > 1:
                                logger.info(f"    {modality}: {len(series_list)} candidates")
                                for series in series_list:
                                    logger.info(f"      - {series.uid}: '{series.protocol_name}'")
                            else:
                                series = series_list[0]
                                logger.info(f"    {modality}: '{series.protocol_name}' ({len(series.files)} files)")
            else:
                # Phase 2: Organize to BIDS with enhanced processing
                organizer = BidsOrganizer(args.output_dir, args.action, max_workers=args.max_workers, streaming_mode=args.streaming_mode)
                organizer.organize_to_bids(collected_data)
            
            elapsed = time.perf_counter() - start_time
            logger.info("=" * 70)
            logger.info(f"Script completed in {elapsed:.2f} seconds")
            logger.info("=" * 70)
            logger.info("="*70)
            logger.info("Enhanced DICOM to BIDS Conversion Completed Successfully")
            logger.info("="*70)
            
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user.")
        print("\nConversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.exception("Full exception details:")
        print(f"Error: Conversion failed. Check log file for details: {args.log_file}")
        sys.exit(1)

    finally:
        # Генерация отчета о производительности
        if args.profile:
            profiler.memory_checkpoint("final")
            report_path = profiler.save_report()
            if report_path:
                print(f"\nPerformance report saved to: {report_path}")
                print(f"Text summary: {report_path.with_suffix('.txt')}")

if __name__ == "__main__":
    main()