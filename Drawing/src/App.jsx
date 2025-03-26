import { useRef, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

const numPixels = 28
const initialPixels = new Array(numPixels).fill(new Array(numPixels).fill(0))

function App() {

  const [pixels, setPixels] = useState(initialPixels)
  const [isDrawing, setIsDrawing] = useState(false)
  const canvasRef = useRef("canvasRef")

  const handleCanvasClick = (e) => {
    if (!isDrawing) return
    draw(e)
  }

  const draw = (e) => {
    console.log(e)
  }

  return (
    <>
      <canvas ref={canvasRef} onMouseMove={handleCanvasClick} onMouseUp={()=>setIsDrawing(false)} onMouseLeave={()=>setIsDrawing(false)} onMouseDown={(e)=>{setIsDrawing(true); draw(e)}} height="700" width="700"/>
      <div style={{display: "flex", flexDirection: "row"}}>
        <button>Reset</button>
        <button>Predict</button>
      </div>
    </>
  )
}

export default App
