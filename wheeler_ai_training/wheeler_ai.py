"""
Wheeler AI - A Non-Transformer Artificial Intelligence
=======================================================

This is an experimental AI architecture that uses Wheeler Memory
(cellular automata dynamics) instead of attention mechanisms.

Core idea:
- Text encodes to spatial patterns in a 2D frame
- Wheeler dynamics create attractors (stable representations)
- Reasoning = manipulating patterns
- Output = decoding patterns back to text

No transformers. No attention. No billions of parameters.
Just dynamics.
"""

import numpy as np
import hashlib
import time
import threading
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import os

# Add local directory to path to find wheeler_cpu if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from wheeler_cpu import WheelerMemoryCPU, NUMBA_AVAILABLE
except ImportError:
    # Fallback if file not found (create dummy)
    print("Warning: wheeler_cpu.py not found. Using dummy implementation.")
    NUMBA_AVAILABLE = False
    
    @dataclass
    class Trajectory:
        final_frame: np.ndarray

    class WheelerMemoryCPU:
        def __init__(self, width, height, use_numba=False):
            self.width = width
            self.height = height
        
        def run_dynamics(self, input_data, max_ticks=50):
            # Dummy implementation
            if isinstance(input_data, bytes):
                frame = np.zeros((self.height, self.width), dtype=np.float32)
            else:
                frame = input_data
            return Trajectory(final_frame=frame)

# =============================================================================
# TEXT CODEC - Converts text ↔ Wheeler frames
# =============================================================================

class TextCodec:
    """
    Encodes text into Wheeler frames and decodes frames back to text.
    
    Encoding strategy:
    - Each character gets a spatial "zone" in the frame
    - Character value determines the pattern seeded in that zone
    - Position in text determines zone location (sequential or 2D)
    - Wheeler dynamics blend overlapping zones into coherent patterns
    
    Decoding strategy:
    - Compare frame regions against known character patterns
    - Use superposition/interference to find closest matches
    - Sequence reconstruction from spatial positions
    """
    
    def __init__(self, width: int = 128, height: int = 128, max_sequence: int = 256):
        self.width = width
        self.height = height
        self.max_sequence = max_sequence
        
        # Character set (ASCII printable + common unicode)
        self.charset = (
            " !\"#$%&'()*+,-./0123456789:;<=>?@"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
            "abcdefghijklmnopqrstuvwxyz{|}~"
            "\n\t"
        )
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.idx_to_char = {i: c for i, c in enumerate(self.charset)}
        self.vocab_size = len(self.charset)
        
        # Precompute character patterns
        self._build_char_patterns()
        
        # Zone layout: how text positions map to frame regions
        self.zone_width = width // 16   # 16 chars per row
        self.zone_height = height // 16  # 16 rows
        self.chars_per_row = width // self.zone_width
        self.total_rows = height // self.zone_height
        
    def _build_char_patterns(self):
        """
        Precompute a unique pattern for each character.
        These patterns will be stamped into zones during encoding.
        """
        self.char_patterns = {}
        pattern_size = 8  # 8x8 pattern per character
        
        for char, idx in self.char_to_idx.items():
            # Create deterministic pattern from character
            seed = hash(f"char_{char}_{idx}") & 0xFFFFFFFF
            rng = np.random.Generator(np.random.PCG64(seed))
            pattern = rng.uniform(-1, 1, (pattern_size, pattern_size)).astype(np.float32)
            self.char_patterns[char] = pattern
    
    def _get_zone(self, position: int) -> Tuple[int, int, int, int]:
        """Get the (y1, y2, x1, x2) bounds for a text position."""
        row = position // self.chars_per_row
        col = position % self.chars_per_row
        
        y1 = row * self.zone_height
        y2 = y1 + self.zone_height
        x1 = col * self.zone_width
        x2 = x1 + self.zone_width
        
        return y1, y2, x1, x2
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a Wheeler frame.
        
        Each character stamps its pattern into a zone.
        Overlapping influences create interference patterns.
        """
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Truncate to max sequence
        text = text[:self.max_sequence]
        
        for pos, char in enumerate(text):
            if char not in self.char_patterns:
                char = ' '  # Unknown char -> space
            
            pattern = self.char_patterns[char]
            y1, y2, x1, x2 = self._get_zone(pos)
            
            # Ensure bounds
            if y2 > self.height or x2 > self.width:
                break
            
            # Stamp pattern into zone (with tiling if zone > pattern)
            zone_h = y2 - y1
            zone_w = x2 - x1
            
            # Tile the pattern to fill the zone
            tiled = np.tile(pattern, (
                (zone_h + 7) // 8,
                (zone_w + 7) // 8
            ))[:zone_h, :zone_w]
            
            # Add to frame (interference)
            frame[y1:y2, x1:x2] += tiled
        
        # Normalize to -1, 1 range
        if frame.max() > 0 or frame.min() < 0:
            max_abs = max(abs(frame.max()), abs(frame.min()))
            frame = frame / max_abs
        
        return frame
    
    def decode(self, frame: np.ndarray, max_length: int = 256) -> str:
        """
        Decode a Wheeler frame back to text.
        
        For each zone, find the character whose pattern best matches.
        """
        result = []
        
        for pos in range(min(max_length, self.chars_per_row * self.total_rows)):
            y1, y2, x1, x2 = self._get_zone(pos)
            
            if y2 > self.height or x2 > self.width:
                break
            
            zone = frame[y1:y2, x1:x2]
            
            # Find best matching character
            best_char = ' '
            best_score = -float('inf')
            
            for char, pattern in self.char_patterns.items():
                # Tile pattern to zone size
                zone_h, zone_w = zone.shape
                tiled = np.tile(pattern, (
                    (zone_h + 7) // 8,
                    (zone_w + 7) // 8
                ))[:zone_h, :zone_w]
                
                # Correlation score
                score = np.sum(zone * tiled)
                
                if score > best_score:
                    best_score = score
                    best_char = char
            
            # Stop at first "empty" zone (low energy)
            zone_energy = np.abs(zone).mean()
            if zone_energy < 0.1:
                break
            
            result.append(best_char)
        
        return ''.join(result).rstrip()


# =============================================================================
# REASONING ENGINE - Pattern manipulation for "thinking"
# =============================================================================

class ReasoningEngine:
    """
    Performs reasoning by manipulating Wheeler frames.
    
    Operations:
    - BLEND: Superimpose multiple frames (association)
    - CONTRAST: Find differences between frames
    - AMPLIFY: Strengthen dominant patterns
    - QUERY: Find resonant memories
    - TRANSFORM: Apply learned transformations
    """
    
    def __init__(self, width: int = 128, height: int = 128):
        self.width = width
        self.height = height
        
        # Learned transformations (pattern -> pattern mappings)
        self.transforms: Dict[str, np.ndarray] = {}
        
    def blend(self, frames: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """Superimpose frames with optional weights."""
        if not frames:
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        if weights is None:
            weights = [1.0] * len(frames)
        
        total = sum(weights)
        weights = [w / total for w in weights]
        
        result = np.zeros_like(frames[0])
        for frame, weight in zip(frames, weights):
            result += frame * weight
        
        return np.clip(result, -1, 1)
    
    def contrast(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        """Find what's in A but not in B."""
        diff = frame_a - frame_b
        return np.clip(diff, -1, 1)
    
    def amplify(self, frame: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """Strengthen dominant patterns, suppress weak ones."""
        # Non-linear amplification
        amplified = np.sign(frame) * (np.abs(frame) ** (1/strength))
        return np.clip(amplified, -1, 1)
    
    def similarity(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Compute similarity between frames (correlation)."""
        correlation = np.sum(frame_a * frame_b) / (self.width * self.height)
        return float((correlation + 1) / 2)  # Normalize to 0-1
    
    def learn_transform(self, name: str, input_frame: np.ndarray, output_frame: np.ndarray):
        """Learn a transformation from input->output pattern."""
        # Simple: store the difference as the transform
        self.transforms[name] = output_frame - input_frame
    
    def apply_transform(self, name: str, frame: np.ndarray) -> np.ndarray:
        """Apply a learned transformation."""
        if name not in self.transforms:
            return frame
        return np.clip(frame + self.transforms[name], -1, 1)


# =============================================================================
# KNOWLEDGE STORE - Wheeler Memory with text metadata
# =============================================================================

@dataclass
class Memory:
    """A memory with both frame representation and text metadata."""
    frame: np.ndarray
    text: str
    timestamp: float = field(default_factory=time.time)
    hits: int = 1
    associations: List[int] = field(default_factory=list)  # Indices of related memories
    
    # Confidence Metadata
    reinforcement_diversity: int = 1  # Number of unique contexts in which this was reinforced
    connectivity: int = 0             # Number of incoming/outgoing associations
    stability: float = 1.0            # Resistance to drift (0.0 to 1.0)
    creation_time: float = field(default_factory=time.time)
    
    @property
    def confidence(self) -> float:
        """
        Calculate dynamic confidence score (0.0 to 1.0).
        High confidence = High hits + Diverse reinforcement + High connectivity + Stability
        """
        # Sigmoid-like normalization for counts
        hit_score = 1 - (1 / (1 + 0.1 * self.hits))
        div_score = 1 - (1 / (1 + 0.2 * self.reinforcement_diversity))
        conn_score = 1 - (1 / (1 + 0.1 * self.connectivity))
        
        # Weighted average
        raw_score = (0.4 * hit_score) + (0.3 * div_score) + (0.2 * conn_score) + (0.1 * self.stability)
        return float(min(1.0, max(0.0, raw_score)))


class KnowledgeStore:
    """
    Long-term knowledge storage using Wheeler Memory.
    
    Stores text+frame pairs with:
    - Associative recall (find related memories)
    - Hit counting (frequently accessed = important)
    - Temporal decay (old unused memories fade)
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories: List[Memory] = []
        self.text_index: Dict[str, int] = {}  # text hash -> index
        
    def store(self, text: str, frame: np.ndarray, context_hash: Optional[str] = None) -> int:
        """Store a memory, return its index."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Check for duplicate
        if text_hash in self.text_index:
            idx = self.text_index[text_hash]
            mem = self.memories[idx]
            mem.hits += 1
            mem.timestamp = time.time()
            
            # Update diversity if context is new (simple heuristic: probabilistic or hash set?)
            # For simplicity, we assume if context_hash is provided and different, we increment
            if context_hash:
                 # In a real system we'd store a set of context hashes, 
                 # but for efficiency we'll just probabilistic increment or increment if explicitly passed
                 mem.reinforcement_diversity += 1
            
            return idx
        
        # Evict if full
        if len(self.memories) >= self.capacity:
            self._evict()
        
        # Store new
        memory = Memory(frame=frame.copy(), text=text)
        if context_hash:
            memory.reinforcement_diversity = 1
            
        idx = len(self.memories)
        self.memories.append(memory)
        self.text_index[text_hash] = idx
        
        return idx
    
    def recall(self, query_frame: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Memory]]:
        """Find memories most similar to query frame."""
        if not self.memories:
            return []
        
        scores = []
        for idx, memory in enumerate(self.memories):
            similarity = np.sum(query_frame * memory.frame) / query_frame.size
            similarity = (similarity + 1) / 2  # Normalize to 0-1
            scores.append((idx, similarity, memory))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update hit counts for recalled memories
        for idx, _, _ in scores[:top_k]:
            self.memories[idx].hits += 1
        
        return scores[:top_k]
    
    def associate(self, idx_a: int, idx_b: int):
        """Create bidirectional association between memories."""
        if 0 <= idx_a < len(self.memories) and 0 <= idx_b < len(self.memories):
            if idx_b not in self.memories[idx_a].associations:
                self.memories[idx_a].associations.append(idx_b)
                self.memories[idx_a].connectivity += 1
            if idx_a not in self.memories[idx_b].associations:
                self.memories[idx_b].associations.append(idx_a)
                self.memories[idx_b].connectivity += 1
    
    def get_associated(self, idx: int) -> List[Memory]:
        """Get memories associated with the given one."""
        if 0 <= idx < len(self.memories):
            return [self.memories[i] for i in self.memories[idx].associations 
                    if i < len(self.memories)]
        return []
    
    def _evict(self):
        """Remove lowest-value memory."""
        if not self.memories:
            return
        
        # Score by hits and recency
        now = time.time()
        scores = []
        for idx, m in enumerate(self.memories):
            age = now - m.timestamp
            score = m.hits / (1 + age / 3600)  # Decay over hours
            scores.append((idx, score))
        
        # Remove lowest scoring
        scores.sort(key=lambda x: x[1])
        victim_idx = scores[0][0]
        
        # Update text index
        victim_hash = hashlib.sha256(self.memories[victim_idx].text.encode()).hexdigest()[:16]
        if victim_hash in self.text_index:
            del self.text_index[victim_hash]
        
        self.memories.pop(victim_idx)
        
        # Reindex
        self.text_index = {}
        for idx, m in enumerate(self.memories):
            h = hashlib.sha256(m.text.encode()).hexdigest()[:16]
            self.text_index[h] = idx


# =============================================================================
# AUTONOMIC SYSTEM - Background processing loop
# =============================================================================

class AutonomicSystem:
    """
    Background autonomous processing.
    
    Functions:
    - Monitor: Watch for patterns that need attention
    - Consolidate: Strengthen frequently-used associations
    - Dream: Random exploration of memory space
    - Alert: Notify when important patterns emerge
    """
    
    def __init__(self, knowledge: KnowledgeStore, reasoning: ReasoningEngine):
        self.knowledge = knowledge
        self.reasoning = reasoning
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_alert: Optional[Callable[[str], None]] = None
        
        # Internal state
        self.attention_queue: List[str] = []
        self.consolidation_count = 0
        
    def start(self):
        """Start background processing."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop background processing."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _run_loop(self):
        """Main autonomic loop."""
        tick = 0
        while self.running:
            time.sleep(1)  # 1 Hz tick rate
            tick += 1
            
            # Periodic consolidation
            if tick % 60 == 0:  # Every minute
                self._consolidate()
            
            # Periodic dreaming (random association exploration)
            if tick % 30 == 0:  # Every 30 seconds
                self._dream()
    
    def _consolidate(self):
        """Strengthen frequently co-accessed memories."""
        self.consolidation_count += 1
        
        # Find high-hit memories
        hot_memories = [m for m in self.knowledge.memories if m.hits > 3]
        
        # Create associations between temporally close hot memories
        for i, m1 in enumerate(hot_memories):
            for m2 in hot_memories[i+1:]:
                time_diff = abs(m1.timestamp - m2.timestamp)
                if time_diff < 300:  # Within 5 minutes
                    idx1 = self.knowledge.memories.index(m1)
                    idx2 = self.knowledge.memories.index(m2)
                    self.knowledge.associate(idx1, idx2)
    
    def _dream(self):
        """Random exploration - blend random memories."""
        if len(self.knowledge.memories) < 2:
            return
        
        # Pick random memories
        indices = np.random.choice(len(self.knowledge.memories), 
                                   size=min(3, len(self.knowledge.memories)), 
                                   replace=False)
        frames = [self.knowledge.memories[i].frame for i in indices]
        
        # Blend them
        dream_frame = self.reasoning.blend(frames)
        
        # Look for resonance with other memories
        matches = self.knowledge.recall(dream_frame, top_k=1)
        
        if matches and matches[0][1] > 0.8:  # High resonance
            # This dream connected to something - maybe interesting?
            pass  # Future: could trigger alerts or insights


# =============================================================================
# WHEELER AI - Main interface
# =============================================================================

class WheelerAI:
    """
    Main Wheeler AI system.
    
    A non-transformer AI that uses cellular automata dynamics
    for memory, reasoning, and generation.
    """
    
    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        use_numba: bool = True
    ):
        self.width = width
        self.height = height
        
        # Core components
        self.dynamics = WheelerMemoryCPU(
            width=width,
            height=height,
            use_numba=use_numba
        )
        self.codec = TextCodec(width=width, height=height)
        self.reasoning = ReasoningEngine(width=width, height=height)
        self.knowledge = KnowledgeStore(capacity=1000)
        self.autonomic = AutonomicSystem(self.knowledge, self.reasoning)
        
        # Conversation state
        self.context_frames: List[np.ndarray] = []  # Recent context
        self.max_context = 10
        
        # Response patterns (learned from usage)
        self.response_patterns: Dict[str, np.ndarray] = {}
        
        print(f"Wheeler AI initialized ({width}x{height} frames)", file=sys.stderr)
        print(f"Numba: {NUMBA_AVAILABLE}", file=sys.stderr)
    
    def process(self, user_input: str) -> str:
        """
        Process user input and generate response.
        
        Pipeline:
        1. Encode input to frame
        2. Run Wheeler dynamics (find attractor)
        3. Query knowledge for relevant memories
        4. Blend context + memories + input
        5. Reason about combined pattern
        6. Decode to text response
        """
        start_time = time.perf_counter()
        
        # 1. Encode input
        input_frame = self.codec.encode(user_input)
        
        # 2. Run dynamics to find stable representation
        traj = self.dynamics.run_dynamics(
            input_frame,
            max_ticks=50
        )
        input_attractor = traj.final_frame
        
        # 3. Query knowledge store
        memories = self.knowledge.recall(input_attractor, top_k=3)
        
        # 4. Build context frame
        context_parts = [input_attractor]
        weights = [2.0]  # Input weighted highest
        
        for _, similarity, memory in memories:
            context_parts.append(memory.frame)
            weights.append(similarity)
        
        # Add recent conversation context
        for ctx_frame in self.context_frames[-3:]:
            context_parts.append(ctx_frame)
            weights.append(0.5)
        
        combined = self.reasoning.blend(context_parts, weights)
        
        # 5. "Reasoning" - amplify dominant patterns
        reasoned = self.reasoning.amplify(combined, strength=1.3)
        
        # 6. Generate response
        # For now: echo back understood content + associated memories
        response_parts = []
        
        # Decode what we understood
        understood = self.codec.decode(reasoned, max_length=100)
        
        # Build response based on what we found
        if memories and memories[0][1] > 0.5:
            # Found relevant memory
            best_memory = memories[0][2]
            response_parts.append(f"[recall: {best_memory.text[:50]}...]")
        
        if understood.strip():
            response_parts.append(f"[understood: {understood[:50]}]")
        
        # Placeholder response generation
        # TODO: This needs actual learned response patterns
        response = self._generate_response(user_input, memories, understood)
        
        # Tension Check & Storage
        tension_detected = False
        if memories:
            idx, score, existing_mem = memories[0]
            # Debug
            # print(f"DEBUG: Score={score:.4f}, Conf={existing_mem.confidence:.4f}")
            
            # If very similar but not identical, and existing memory is high confidence
            # Lowered threshold to 0.70 based on empirical testing with TextCodec
            if 0.70 < score < 0.99:
                if existing_mem.confidence > 0.60:
                    # Potential tension/contradiction with established belief
                    tension_detected = True
                    msg = f"\n[!] Tension detected with established belief: '{existing_mem.text}' (Conf: {existing_mem.confidence:.2f})"
                    response += msg
                    # We do NOT store strictly contradictory/tension-causing inputs immediately
                    # This implements the "Epistemological Independence" - don't just overwrite.

        if not tension_detected:
            # Store this interaction
            # Generate context hash from recent frames for diversity tracking
            if self.context_frames:
                ctx_bytes = b"".join([f.tobytes() for f in self.context_frames[-3:]])
                ctx_hash = hashlib.sha256(ctx_bytes).hexdigest()[:16]
            else:
                ctx_hash = None
                
            self.knowledge.store(user_input, input_attractor, context_hash=ctx_hash)
        
        # Update context
        self.context_frames.append(input_attractor)
        if len(self.context_frames) > self.max_context:
            self.context_frames.pop(0)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return f"{response}\n[{elapsed:.1f}ms]"
    
    def _generate_response(
        self, 
        user_input: str, 
        memories: List[Tuple[int, float, Memory]],
        understood: str
    ) -> str:
        """
        Generate a response based on input and retrieved memories.
        
        Current implementation: Pattern matching + templates
        Future: Learned generation from Wheeler dynamics
        """
        user_lower = user_input.lower().strip()
        
        # Handle greetings
        if any(g in user_lower for g in ['hello', 'hi', 'hey']):
            return "Hello! I'm Wheeler AI - a non-transformer experimental system."
        
        # Handle questions about self
        if 'what are you' in user_lower or 'who are you' in user_lower:
            return ("I'm an experimental AI using Wheeler Memory dynamics instead of "
                   "transformers. I store patterns, find associations, and generate "
                   "responses through cellular automata - not attention mechanisms.")
        
        if 'how do you work' in user_lower:
            return ("I encode text into 2D patterns, run agar.io-style dynamics to "
                   "find stable attractors, query similar patterns from memory, "
                   "blend them together, and decode back to text. No neural networks!")
        
        # Handle memory queries
        if 'remember' in user_lower or 'recall' in user_lower:
            if memories:
                mem_texts = [m.text for _, _, m in memories[:3]]
                return f"I remember: {'; '.join(mem_texts)}"
            return "I don't have relevant memories yet. Tell me more!"
        
        # Handle teaching
        if user_lower.startswith('learn:') or user_lower.startswith('remember:'):
            content = user_input.split(':', 1)[1].strip()
            return f"Stored in memory: '{content[:50]}...'"
        
        # Default: acknowledge and invite more
        if memories and memories[0][1] > 0.3:
            related = memories[0][2].text[:30]
            return f"That reminds me of: '{related}...' - Tell me more?"
        
        return f"I'm learning. Tell me more about '{user_input[:20]}...'"
    
    def start_autonomic(self):
        """Start background autonomous processing."""
        self.autonomic.start()
        print("Autonomic system started", file=sys.stderr)
    
    def stop_autonomic(self):
        """Stop background autonomous processing."""
        self.autonomic.stop()
        print("Autonomic system stopped", file=sys.stderr)
    
    def stats(self) -> Dict:
        """Get system statistics."""
        avg_conf = 0.0
        if self.knowledge.memories:
            avg_conf = sum(m.confidence for m in self.knowledge.memories) / len(self.knowledge.memories)
            
        return {
            'memories': len(self.knowledge.memories),
            'avg_confidence': round(avg_conf, 3),
            'context_depth': len(self.context_frames),
            'response_patterns': len(self.response_patterns),
            'autonomic_consolidations': self.autonomic.consolidation_count
        }


# =============================================================================
# INTERACTIVE SHELL
# =============================================================================

def interactive_shell():
    """Run interactive Wheeler AI shell."""
    print("=" * 60)
    print("WHEELER AI - Non-Transformer Experimental System")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /stats  - Show system statistics")
    print("  /auto   - Toggle autonomic system")
    print("  /clear  - Clear conversation context")
    print("  /quit   - Exit")
    print()
    
    ai = WheelerAI(width=128, height=128)
    autonomic_running = False
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith('/'):
            cmd = user_input.lower()
            
            if cmd == '/quit':
                break
            elif cmd == '/stats':
                print(f"Stats: {ai.stats()}")
                continue
            elif cmd == '/auto':
                if autonomic_running:
                    ai.stop_autonomic()
                    autonomic_running = False
                else:
                    ai.start_autonomic()
                    autonomic_running = True
                continue
            elif cmd == '/clear':
                ai.context_frames = []
                print("Context cleared.")
                continue
            elif cmd == '/memory':
                # Show top memories by confidence
                sorted_mem = sorted(ai.knowledge.memories, key=lambda m: m.confidence, reverse=True)
                print("\nTop Beliefs (by Confidence):")
                for i, m in enumerate(sorted_mem[:5]):
                    print(f" {i+1}. [{m.confidence:.2f}] {m.text[:60]} (Hits: {m.hits}, Div: {m.reinforcement_diversity})")
                print()
                continue
        
        # Process input
        response = ai.process(user_input)
        print(f"Wheeler: {response}")
        print()
    
    if autonomic_running:
        ai.stop_autonomic()
    
    print("\nGoodbye!")


if __name__ == "__main__":
    interactive_shell()
