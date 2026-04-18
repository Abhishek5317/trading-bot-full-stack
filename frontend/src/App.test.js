import { render, screen } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      json: () => Promise.resolve({ status: 'ACTIVE' })
    })
  );
});

afterEach(() => {
  jest.resetAllMocks();
});

test('renders dashboard title', async () => {
  render(<App />);
  expect(await screen.findByText(/Trading Bot Dashboard/i)).toBeInTheDocument();
});
