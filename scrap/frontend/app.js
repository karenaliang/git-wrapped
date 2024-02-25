import logo from './logo.svg';
import './App.css';

import axios from 'axios';

// function fetchAPI() {
//   axios.get('http://localhost:5000/hello')
//     .then(response => console.log(response.data))
// }

// class App extends React.Component {
//     componentDidMount() {
//       fetchAPI();
//     }
  
//     render() {
//       return (
//         // render code here
//         <div>
//         <h1>Hello, React App!</h1>
//         <p>This is my React application.</p>
//         {/* Other UI components and content */}
//         </div>

//       );
//     }
//   }
  

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Output:
        </p>
      </header>
    </div>
  );
}

export default App;
