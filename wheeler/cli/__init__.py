import click
import asyncio
import os
from wheeler.core.memory import WheelerMemory
from wheeler.core.viz import render_frame

def async_command(f):
    import functools
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@click.group()
def main():
    """Wheeler Memory CLI - Biological-grade associative memory."""
    pass

@main.command()
@click.argument('text')
@click.option('--storage', envvar='WHEELER_STORAGE', default='./.wheeler', help='Storage directory.')
@async_command
async def store(text, storage):
    """Store text into Wheeler Memory."""
    os.makedirs(storage, exist_ok=True)
    wm = WheelerMemory(storage)
    await wm.initialize()
    
    click.echo(f"Encoding and running dynamics for: '{text}'...")
    memory_uuid = await wm.store(text)
    click.echo(f"Stored memory successfully.")
    click.echo(f"UUID: {memory_uuid}")

@main.command()
@click.argument('text')
@click.option('--storage', envvar='WHEELER_STORAGE', default='./.wheeler', help='Storage directory.')
@click.option('--limit', default=3, help='Number of results to return.')
@async_command
async def recall(text, storage, limit):
    """Recall similar memories from Wheeler Memory."""
    if not os.path.exists(storage):
        click.echo(f"Storage directory {storage} does not exist.")
        return

    wm = WheelerMemory(storage)
    await wm.initialize()
    
    click.echo(f"Querying: '{text}'...")
    results = await wm.recall(text, limit=limit)
    
    if not results:
        click.echo("No memories found.")
        return

    click.echo("\nResults:")
    for i, res in enumerate(results):
        click.echo(f"{i+1}. Key: {res['key']}")
        click.echo(f"   Similarity: {res['similarity']:.4f}")
        click.echo(f"   Stability: {res['stability']:.4f}")
        click.echo(f"   Hits: {res['hit_count']}")
        click.echo("-" * 20)

@main.command()
@click.argument('uuid')
@click.option('--output', '-o', default=None, help='Output image path.')
@click.option('--storage', envvar='WHEELER_STORAGE', default='./.wheeler', help='Storage directory.')
@async_command
async def viz(uuid, output, storage):
    """Visualize a memory frame by UUID."""
    if not os.path.exists(storage):
        click.echo(f"Storage directory {storage} does not exist.")
        return

    wm = WheelerMemory(storage)
    await wm.initialize()
    
    click.echo(f"Loading memory {uuid}...")
    memory = await wm.load_by_uuid(uuid)
    
    if not memory:
        click.echo(f"No memory found with UUID: {uuid}")
        return
        
    if not output:
        output = f"frame_{uuid[:8]}.png"
        
    click.echo(f"Rendering frame to {output}...")
    render_frame(memory["frame"], output, title=f"Memory: {memory['key']}")
    click.echo("Done.")

@main.command()
@click.argument('text')
@click.option('--output', '-o', default='evolution.gif', help='Output GIF path.')
@click.option('--steps', default=30, help='Number of dynamics steps.')
@click.option('--fps', default=10, help='Frames per second.')
@async_command
async def viz_run(text, output, steps, fps):
    """Visualize the evolution of a text input into an attractor."""
    from wheeler.core.codec import TextCodec
    from wheeler.core.engine import DynamicsEngine
    from wheeler.core.viz import animate_trajectory
    
    click.echo(f"Encoding '{text}'...")
    codec = TextCodec()
    initial_grid = codec.encode(text)
    
    engine = DynamicsEngine()
    click.echo(f"Running dynamics for {steps} steps...")
    trajectory = engine.run_trajectory(initial_grid, steps=steps)
    
    click.echo(f"Animating trajectory to {output}...")
    animate_trajectory(trajectory, output, fps=fps)
    click.echo("Done.")

@main.command()
@click.argument('text_a')
@click.argument('text_b')
@click.option('--storage', envvar='WHEELER_STORAGE', default='./.wheeler', help='Storage directory.')
@click.option('--limit', default=3, help='Number of results to return.')
@async_command
async def reason(text_a, text_b, storage, limit):
    """Blend two concepts and find the resulting attractor."""
    if not os.path.exists(storage):
        click.echo(f"Storage directory {storage} does not exist.")
        return

    wm = WheelerMemory(storage)
    await wm.initialize()
    
    click.echo(f"Reasoning between: '{text_a}' and '{text_b}'...")
    results = await wm.infer(text_a, text_b, limit=limit)
    
    if not results:
        click.echo("No associated memories found for this inference.")
        return

    click.echo("\nInferred Associations:")
    for i, res in enumerate(results):
        click.echo(f"{i+1}. Key: {res['key']}")
        click.echo(f"   Score: {res['score']:.4f}")
        click.echo("-" * 20)

@main.command()
@click.option('--storage', envvar='WHEELER_STORAGE', default='./.wheeler', help='Storage directory.')
@click.option('--ticks', default=5, help='Number of autonomic ticks to run.')
@async_command
async def dream(storage, ticks):
    """Run autonomic background processes (dreaming, consolidation)."""
    from wheeler.core.autonomic import AutonomicSystem
    
    if not os.path.exists(storage):
        click.echo(f"Storage directory {storage} does not exist.")
        return

    wm = WheelerMemory(storage)
    await wm.initialize()
    
    system = AutonomicSystem(wm)
    system.tick_rate = 1.0 # Fast for CLI foreground run
    
    click.echo(f"Starting autonomic processes for {ticks} ticks...")
    await system.start()
    
    for i in range(ticks):
        await asyncio.sleep(system.tick_rate)
        click.echo(f"Tick {i+1}/{ticks}...")
        
    await system.stop()
    click.echo("Autonomic cycle complete.")

@main.command()
@click.option('--storage', envvar='WHEELER_STORAGE', default='./.wheeler', help='Storage directory.')
@click.option('--port', default=5000, help='Port to run dashboard on.')
def dashboard(storage, port):
    """Launch the Web Dashboard."""
    try:
        from wheeler.web.app import app
        import os
        
        # Set environment variable for app to read
        os.environ["WHEELER_STORAGE"] = storage
        
        click.echo(f"Starting dashboard on http://127.0.0.1:{port}")
        app.run(host='127.0.0.1', port=port)
    except ImportError as e:
        click.echo(f"Error importing dashboard: {e}")
        click.echo("Make sure flask is installed: pip install flask")

if __name__ == "__main__":
    main()
