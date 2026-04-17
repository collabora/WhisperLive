"""
Optional speaker diarization module for WhisperLive.

Uses speaker embeddings and online clustering to assign speaker labels
to transcription segments in real-time. Requires pyannote.audio as an
optional dependency.

Install: pip install pyannote.audio
"""

import logging
import numpy as np


class SpeakerDiarizer:
    """Real-time speaker diarization using speaker embeddings and online clustering.

    Each completed transcription segment's audio is passed through a speaker
    embedding model. The embedding is compared against known speakers using
    cosine similarity. If no match exceeds the threshold, a new speaker is
    created.

    Args:
        similarity_threshold (float): Minimum cosine similarity to match an
            existing speaker. Lower values merge speakers more aggressively.
            Default 0.55.
        max_speakers (int): Maximum number of distinct speakers to track.
            Once reached, new segments are assigned to the closest existing
            speaker. Default 10.
        embedding_model (str): The pyannote embedding model to use.
            Default "pyannote/wespeaker-voxceleb-resnet34-LM".
        hf_token (str or None): HuggingFace token for gated model access.
    """

    def __init__(
        self,
        similarity_threshold=0.55,
        max_speakers=10,
        embedding_model="pyannote/wespeaker-voxceleb-resnet34-LM",
        hf_token=None,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.speakers = {}  # speaker_id -> embedding (averaged)
        self._speaker_count = 0
        self._model = None
        self._embedding_model_name = embedding_model
        self._hf_token = hf_token

    def _load_model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is not None:
            return
        try:
            from pyannote.audio import Model, Inference
            import torch

            model = Model.from_pretrained(
                self._embedding_model_name,
                use_auth_token=self._hf_token,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = Inference(model, window="whole", device=torch.device(device))
            logging.info(f"Speaker embedding model loaded on {device}")
        except ImportError:
            raise ImportError(
                "pyannote.audio is required for speaker diarization. "
                "Install it with: pip install pyannote.audio"
            )

    def _compute_embedding(self, audio_np, sample_rate=16000):
        """Compute a speaker embedding from an audio numpy array.

        Args:
            audio_np (np.ndarray): 1-D float32 audio samples.
            sample_rate (int): Sample rate of the audio.

        Returns:
            np.ndarray: Speaker embedding vector, or None if audio is too short.
        """
        self._load_model()
        if len(audio_np) < sample_rate * 0.3:
            return None
        waveform = {
            "waveform": __import__("torch").tensor(audio_np).unsqueeze(0),
            "sample_rate": sample_rate,
        }
        embedding = self._model(waveform)
        return embedding / np.linalg.norm(embedding)

    @staticmethod
    def _cosine_similarity(a, b):
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b))

    def identify_speaker(self, audio_np, sample_rate=16000):
        """Identify or create a speaker from an audio segment.

        Args:
            audio_np (np.ndarray): 1-D float32 audio for the segment.
            sample_rate (int): Sample rate. Default 16000.

        Returns:
            str or None: Speaker label (e.g. "SPEAKER_00"), or None if
                the audio is too short to embed.
        """
        embedding = self._compute_embedding(audio_np, sample_rate)
        if embedding is None:
            return None

        best_speaker = None
        best_sim = -1.0

        for speaker_id, stored_emb in self.speakers.items():
            sim = self._cosine_similarity(embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_speaker = speaker_id

        if best_sim >= self.similarity_threshold:
            # Update running average for the matched speaker
            self.speakers[best_speaker] = (
                self.speakers[best_speaker] * 0.9 + embedding * 0.1
            )
            # Re-normalize
            self.speakers[best_speaker] /= np.linalg.norm(self.speakers[best_speaker])
            return best_speaker

        if len(self.speakers) >= self.max_speakers:
            # Assign to closest speaker
            return best_speaker if best_speaker else f"SPEAKER_{self._speaker_count:02d}"

        # Create a new speaker
        speaker_id = f"SPEAKER_{self._speaker_count:02d}"
        self._speaker_count += 1
        self.speakers[speaker_id] = embedding
        return speaker_id

    def reset(self):
        """Reset all speaker state."""
        self.speakers.clear()
        self._speaker_count = 0

    def enroll_speaker(self, name, audio_np, sample_rate=16000):
        """Enroll a known speaker from a reference audio clip.

        After enrollment, segments matching this speaker will be labeled
        with the provided name instead of a generic "SPEAKER_XX" label.

        Args:
            name (str): Human-readable name for the speaker (e.g. "Alice").
            audio_np (np.ndarray): 1-D float32 reference audio clip
                (recommended 2-10 seconds).
            sample_rate (int): Sample rate of the reference audio.

        Returns:
            bool: True if enrollment succeeded, False if audio was too short.

        Raises:
            ImportError: If pyannote.audio is not installed.
        """
        embedding = self._compute_embedding(audio_np, sample_rate)
        if embedding is None:
            return False

        self.speakers[name] = embedding
        logging.info(f"Enrolled known speaker: {name}")
        return True

    def enroll_speakers_from_files(self, speaker_refs, sample_rate=16000):
        """Enroll multiple speakers from audio file paths or numpy arrays.

        Args:
            speaker_refs (dict): Mapping of speaker name -> audio data.
                Values can be:
                - np.ndarray: Raw audio array
                - str: File path to audio file
            sample_rate (int): Default sample rate for raw arrays.

        Returns:
            dict: Mapping of speaker name -> success (bool).
        """
        results = {}
        for name, audio in speaker_refs.items():
            if isinstance(audio, str):
                # Load from file path
                try:
                    import soundfile as sf
                    data, sr = sf.read(audio, dtype="float32")
                    if data.ndim > 1:
                        data = data[:, 0]  # Take first channel
                    results[name] = self.enroll_speaker(name, data, sr)
                except Exception as e:
                    logging.error(f"Failed to enroll speaker '{name}' from file: {e}")
                    results[name] = False
            elif isinstance(audio, np.ndarray):
                results[name] = self.enroll_speaker(name, audio, sample_rate)
            else:
                logging.error(f"Unsupported audio type for speaker '{name}': {type(audio)}")
                results[name] = False
        return results

    def get_enrolled_speakers(self):
        """Return list of currently enrolled speaker names."""
        return list(self.speakers.keys())
