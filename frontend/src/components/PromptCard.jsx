import React from 'react';
import '../styles/PromptCard.css'; // Add your CSS styles for the card

const PromptCard = ({ prompt, onClick }) => (
  <div className='prompt-card' onClick={() => onClick(prompt)}>
    <h4 className='prompt-title'>{prompt.title}</h4>
    <p className='prompt-description'>{prompt.description}</p>
  </div>
);

export default PromptCard;
