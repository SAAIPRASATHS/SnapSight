import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from '../src/App';

describe('App', () => {
  it('renders without crashing', () => {
    expect(() => {
      render(
        <MemoryRouter initialEntries={['/dashboard']}>
          <App />
        </MemoryRouter>,
      );
    }).not.toThrow();
  });

  it('displays SnapSight branding text', () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <App />
      </MemoryRouter>,
    );

    const snapSightElements = screen.getAllByText(/SnapSight/i);
    expect(snapSightElements.length).toBeGreaterThan(0);
  });
});
