import logo from './logo.svg';
import './App.css';
import ChartComponent2 from "./components/RowChartDouble";
import ChartComponent from "./components/RowChart";

function App() {
  return (
    <div className="App">
      <header className="App-header">
          <ChartComponent/>
          <ChartComponent2/>
      </header>
    </div>
  );
}

export default App;
