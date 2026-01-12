import websockets
import asyncio

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send("hello")
        response = await websocket.recv()
        assert response == 'msg: hello'
        print("Test passed!")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_websocket())
