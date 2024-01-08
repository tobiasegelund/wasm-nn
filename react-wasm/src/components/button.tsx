export default function Button({value, weight, handleClick}) {
  return (
    <div>
      <button className="buttons-button" onClick={handleClick}>
        {value}
        <div className="button-weight">
          {weight && weight.toFixed(2)}
        </div>
      </button>
    </div>
  )
}