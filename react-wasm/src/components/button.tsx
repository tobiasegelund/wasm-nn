export default function Button({value, handleClick}) {
  return (
    <div>
      <button className="buttons-button" onClick={handleClick}> {value} </button>
    </div>
  )
}