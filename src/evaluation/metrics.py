"""
Metrics calculation for music transcription evaluation.

Provides comprehensive metrics for assessing transcription quality including:
- Note-level accuracy
- Onset/offset F1-scores
- Pitch precision/recall
- Timing deviation metrics
- Tempo accuracy
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import logging


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes:
        note_accuracy: Percentage of correctly transcribed notes
        onset_precision: Precision for note onsets
        onset_recall: Recall for note onsets
        onset_f1: F1-score for note onsets
        offset_precision: Precision for note offsets
        offset_recall: Recall for note offsets
        offset_f1: F1-score for note offsets
        pitch_precision: Precision for pitch detection
        pitch_recall: Recall for pitch detection
        pitch_f1: F1-score for pitch detection
        timing_error_mean: Mean timing error in seconds
        timing_error_std: Standard deviation of timing error
        tempo_error: Relative error in tempo detection
        total_notes_predicted: Number of predicted notes
        total_notes_ground_truth: Number of ground truth notes
    """
    note_accuracy: float = 0.0
    onset_precision: float = 0.0
    onset_recall: float = 0.0
    onset_f1: float = 0.0
    offset_precision: float = 0.0
    offset_recall: float = 0.0
    offset_f1: float = 0.0
    pitch_precision: float = 0.0
    pitch_recall: float = 0.0
    pitch_f1: float = 0.0
    timing_error_mean: float = 0.0
    timing_error_std: float = 0.0
    tempo_error: float = 0.0
    total_notes_predicted: int = 0
    total_notes_ground_truth: int = 0
    
    # Detailed error analysis
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    
    # Per-pitch statistics
    pitch_errors: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'note_accuracy': self.note_accuracy,
            'onset_precision': self.onset_precision,
            'onset_recall': self.onset_recall,
            'onset_f1': self.onset_f1,
            'offset_precision': self.offset_precision,
            'offset_recall': self.offset_recall,
            'offset_f1': self.offset_f1,
            'pitch_precision': self.pitch_precision,
            'pitch_recall': self.pitch_recall,
            'pitch_f1': self.pitch_f1,
            'timing_error_mean': self.timing_error_mean,
            'timing_error_std': self.timing_error_std,
            'tempo_error': self.tempo_error,
            'total_notes_predicted': self.total_notes_predicted,
            'total_notes_ground_truth': self.total_notes_ground_truth,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'true_positives': self.true_positives,
            'pitch_errors': self.pitch_errors,
        }


@dataclass
class Note:
    """Simple note representation for evaluation."""
    pitch: int  # MIDI note number
    onset: float  # Start time in seconds
    offset: float  # End time in seconds
    velocity: int = 64


class MetricCalculator:
    """
    Calculate comprehensive evaluation metrics for transcription.
    
    Compares predicted notes with ground truth using various metrics
    including note accuracy, F1-scores, precision/recall, and timing errors.
    """
    
    def __init__(
        self,
        onset_tolerance: float = 0.05,
        offset_tolerance: float = 0.05,
        pitch_tolerance: int = 0
    ):
        """
        Initialize metric calculator.
        
        Args:
            onset_tolerance: Maximum time difference for onset matching (seconds)
            offset_tolerance: Maximum time difference for offset matching (seconds)
            pitch_tolerance: Maximum semitone difference for pitch matching
        """
        self.onset_tolerance = onset_tolerance
        self.offset_tolerance = offset_tolerance
        self.pitch_tolerance = pitch_tolerance
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(
        self,
        predicted_notes: List[Note],
        ground_truth_notes: List[Note],
        predicted_tempo: Optional[float] = None,
        ground_truth_tempo: Optional[float] = None
    ) -> EvaluationMetrics:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predicted_notes: List of predicted notes
            ground_truth_notes: List of ground truth notes
            predicted_tempo: Predicted tempo in BPM
            ground_truth_tempo: Ground truth tempo in BPM
            
        Returns:
            EvaluationMetrics with all calculated metrics
        """
        metrics = EvaluationMetrics()
        
        metrics.total_notes_predicted = len(predicted_notes)
        metrics.total_notes_ground_truth = len(ground_truth_notes)
        
        if len(ground_truth_notes) == 0:
            self.logger.warning("No ground truth notes provided")
            return metrics
        
        # Match notes between predicted and ground truth
        matches, unmatched_pred, unmatched_gt = self._match_notes(
            predicted_notes, ground_truth_notes
        )
        
        # Calculate note-level accuracy
        metrics.note_accuracy = self._calculate_note_accuracy(
            matches, predicted_notes, ground_truth_notes
        )
        
        # Calculate onset metrics
        onset_tp, onset_fp, onset_fn = self._calculate_onset_metrics(
            predicted_notes, ground_truth_notes, matches
        )
        metrics.onset_precision = self._safe_divide(onset_tp, onset_tp + onset_fp)
        metrics.onset_recall = self._safe_divide(onset_tp, onset_tp + onset_fn)
        metrics.onset_f1 = self._calculate_f1(
            metrics.onset_precision, metrics.onset_recall
        )
        
        # Calculate offset metrics
        offset_tp, offset_fp, offset_fn = self._calculate_offset_metrics(
            predicted_notes, ground_truth_notes, matches
        )
        metrics.offset_precision = self._safe_divide(offset_tp, offset_tp + offset_fp)
        metrics.offset_recall = self._safe_divide(offset_tp, offset_tp + offset_fn)
        metrics.offset_f1 = self._calculate_f1(
            metrics.offset_precision, metrics.offset_recall
        )
        
        # Calculate pitch metrics
        pitch_tp, pitch_fp, pitch_fn = self._calculate_pitch_metrics(
            predicted_notes, ground_truth_notes, matches
        )
        metrics.pitch_precision = self._safe_divide(pitch_tp, pitch_tp + pitch_fp)
        metrics.pitch_recall = self._safe_divide(pitch_tp, pitch_tp + pitch_fn)
        metrics.pitch_f1 = self._calculate_f1(
            metrics.pitch_precision, metrics.pitch_recall
        )
        
        # Store error counts
        metrics.true_positives = len(matches)
        metrics.false_positives = len(unmatched_pred)
        metrics.false_negatives = len(unmatched_gt)
        
        # Calculate timing errors
        if matches:
            timing_errors = [
                abs(pred.onset - gt.onset)
                for pred, gt in matches
            ]
            metrics.timing_error_mean = float(np.mean(timing_errors))
            metrics.timing_error_std = float(np.std(timing_errors))
        
        # Calculate pitch errors
        metrics.pitch_errors = self._calculate_pitch_errors(
            predicted_notes, ground_truth_notes, matches
        )
        
        # Calculate tempo error
        if predicted_tempo is not None and ground_truth_tempo is not None:
            metrics.tempo_error = abs(
                predicted_tempo - ground_truth_tempo
            ) / ground_truth_tempo
        
        return metrics
    
    def _match_notes(
        self,
        predicted: List[Note],
        ground_truth: List[Note]
    ) -> Tuple[List[Tuple[Note, Note]], List[Note], List[Note]]:
        """
        Match predicted notes to ground truth notes.
        
        Uses greedy matching based on onset time and pitch.
        
        Returns:
            Tuple of (matches, unmatched_predicted, unmatched_ground_truth)
        """
        matches = []
        unmatched_pred = []
        unmatched_gt = list(ground_truth)
        
        for pred_note in predicted:
            best_match = None
            best_distance = float('inf')
            
            for gt_note in unmatched_gt:
                # Check if notes match within tolerances
                onset_diff = abs(pred_note.onset - gt_note.onset)
                pitch_diff = abs(pred_note.pitch - gt_note.pitch)
                
                if (onset_diff <= self.onset_tolerance and
                    pitch_diff <= self.pitch_tolerance):
                    
                    # Calculate combined distance for best match
                    distance = onset_diff + pitch_diff * 0.1
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = gt_note
            
            if best_match is not None:
                matches.append((pred_note, best_match))
                unmatched_gt.remove(best_match)
            else:
                unmatched_pred.append(pred_note)
        
        return matches, unmatched_pred, unmatched_gt
    
    def _calculate_note_accuracy(
        self,
        matches: List[Tuple[Note, Note]],
        predicted: List[Note],
        ground_truth: List[Note]
    ) -> float:
        """Calculate note-level accuracy."""
        if len(ground_truth) == 0:
            return 0.0
        
        # Correct notes are those that match in pitch, onset, and offset
        correct = sum(
            1 for pred, gt in matches
            if abs(pred.pitch - gt.pitch) <= self.pitch_tolerance
            and abs(pred.onset - gt.onset) <= self.onset_tolerance
            and abs(pred.offset - gt.offset) <= self.offset_tolerance
        )
        
        return correct / len(ground_truth)
    
    def _calculate_onset_metrics(
        self,
        predicted: List[Note],
        ground_truth: List[Note],
        matches: List[Tuple[Note, Note]]
    ) -> Tuple[int, int, int]:
        """Calculate onset-based TP, FP, FN."""
        # True positives: notes with correct onset (within tolerance)
        tp = sum(
            1 for pred, gt in matches
            if abs(pred.onset - gt.onset) <= self.onset_tolerance
        )
        
        # False positives: predicted notes without matching ground truth
        fp = len(predicted) - len(matches)
        
        # False negatives: ground truth notes without matching prediction
        fn = len(ground_truth) - len(matches)
        
        return tp, fp, fn
    
    def _calculate_offset_metrics(
        self,
        predicted: List[Note],
        ground_truth: List[Note],
        matches: List[Tuple[Note, Note]]
    ) -> Tuple[int, int, int]:
        """Calculate offset-based TP, FP, FN."""
        # True positives: notes with correct offset (within tolerance)
        tp = sum(
            1 for pred, gt in matches
            if abs(pred.offset - gt.offset) <= self.offset_tolerance
        )
        
        fp = len(predicted) - tp
        fn = len(ground_truth) - tp
        
        return tp, fp, fn
    
    def _calculate_pitch_metrics(
        self,
        predicted: List[Note],
        ground_truth: List[Note],
        matches: List[Tuple[Note, Note]]
    ) -> Tuple[int, int, int]:
        """Calculate pitch-based TP, FP, FN."""
        # True positives: notes with correct pitch (within tolerance)
        tp = sum(
            1 for pred, gt in matches
            if abs(pred.pitch - gt.pitch) <= self.pitch_tolerance
        )
        
        fp = len(predicted) - tp
        fn = len(ground_truth) - tp
        
        return tp, fp, fn
    
    def _calculate_pitch_errors(
        self,
        predicted: List[Note],
        ground_truth: List[Note],
        matches: List[Tuple[Note, Note]]
    ) -> Dict[int, int]:
        """Calculate per-pitch error distribution."""
        pitch_errors = defaultdict(int)
        
        # Count errors for each pitch in ground truth
        matched_gt_pitches = {gt.pitch for _, gt in matches}
        
        for gt_note in ground_truth:
            if gt_note.pitch not in matched_gt_pitches:
                pitch_errors[gt_note.pitch] += 1
        
        return dict(pitch_errors)
    
    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Safe division that returns 0 if denominator is 0."""
        return numerator / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def _calculate_f1(precision: float, recall: float) -> float:
        """Calculate F1-score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
