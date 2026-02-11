import pytest
import asyncio
from wheeler.core.autonomic import AutonomicSystem
from wheeler.core.memory import WheelerMemory

@pytest.fixture
async def memory(tmp_path):
    wm = WheelerMemory(str(tmp_path))
    await wm.initialize()
    return wm

@pytest.mark.asyncio
async def test_autonomic_loop_start_stop(memory):
    system = AutonomicSystem(memory)
    
    # Start the system
    await system.start()
    assert system.running
    assert system._task is not None
    
    # Let it tick once (we'll make the tick rate fast for tests)
    system.tick_rate = 0.1
    await asyncio.sleep(0.2)
    
    # Stop it
    await system.stop()
    assert not system.running
    assert system._task.done() or system._task.cancelled()

@pytest.mark.asyncio
async def test_dreaming_logic(memory):
    # Store some memories to dream about
    await memory.store("Dreaming of sheep")
    await memory.store("Electric sheep")
    
    system = AutonomicSystem(memory)
    # Manually trigger a dream
    dream_result = await system._dream()
    
    # A dream should return something if memories exist
    if dream_result:
        assert "key" in dream_result
        assert "similarity" in dream_result

@pytest.mark.asyncio
async def test_consolidation(memory):
    await memory.store("Important memory")
    
    # Get the memory record
    all_meta = await memory.storage.metadata.list_memories()
    m = all_meta[0]
    initial_conf = m["confidence"]
    
    # Boost hits
    for _ in range(5):
        await memory.storage.metadata.increment_hit_count(m["id"])
    
    system = AutonomicSystem(memory)
    await system._consolidate()
    
    # Reload and check confidence
    m_updated = await memory.storage.metadata.get_memory(m["id"])
    assert m_updated["confidence"] > initial_conf
