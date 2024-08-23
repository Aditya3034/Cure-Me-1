import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useLayoutEffect,
} from 'react';
import { BiPlus, BiUser, BiSend, BiSolidUserCircle } from 'react-icons/bi';
import { MdOutlineArrowLeft, MdOutlineArrowRight, MdMenu } from 'react-icons/md';
import axios from 'axios';
import PromptCard from './components/PromptCard.jsx';

const prompts = [
  { title: 'General Inquiry', description: 'Ask anything about our services.' },
  { title: 'Technical Support', description: 'Get help with technical issues.' },
  { title: 'Feedback', description: 'Provide feedback on our services.' },
  // Add more prompts as needed
];

function App() {
  const [text, setText] = useState('');
  const [message, setMessage] = useState(null);
  const [previousChats, setPreviousChats] = useState([]);
  const [localChats, setLocalChats] = useState([]);
  const [currentTitle, setCurrentTitle] = useState(null);
  const [isResponseLoading, setIsResponseLoading] = useState(false);
  const [errorText, setErrorText] = useState('');
  const [isShowSidebar, setIsShowSidebar] = useState(false);
  const scrollToLastItem = useRef(null);

  const createNewChat = () => {
    setMessage(null);
    setText('');
    setCurrentTitle(null);
  };

  const backToHistoryPrompt = (uniqueTitle) => {
    setCurrentTitle(uniqueTitle);
    setMessage(null);
    setText('');
  };

  const toggleSidebar = useCallback(() => {
    setIsShowSidebar((prev) => !prev);
  }, []);

  function formatApiResponse(response) {
    // Replace certain symbols with more readable alternatives
    let formattedResponse = response.replace(/\*/g, ''); // Remove asterisks
    formattedResponse = formattedResponse.replace(/\\n/g, '\n'); // Replace escaped newlines with actual newlines
  
    // Optionally, add bullet points for lists
    formattedResponse = formattedResponse.replace(/- /g, '\n- '); // Ensure bullet points are properly formatted
  
    // Capitalize the first letter of each sentence
    formattedResponse = formattedResponse.replace(/(^\w|\.\s*\w)/g, (c) => c.toUpperCase());
  
    return formatListResponse(formattedResponse.trim());
  }
  function formatListResponse(response) {
    // Replace the numbers followed by periods with a newline and the number
    let formattedResponse = response.replace(/(\d+)\.\s*/g, '\n$1. ');
  
    // Trim any leading or trailing whitespace
    formattedResponse = formattedResponse.trim();
  
    return formattedResponse;
  }
  
  const answers = {
    "I'm feeling anxious all the time. What can I do to manage it?": 
      "Anxiety is a common mental health condition that can be managed with a variety of techniques. Some helpful strategies include:\n\n* *Exercise:* Exercise can help to reduce stress and anxiety levels, and improve mood. Aim for at least 30 minutes of moderate-intensity exercise most days of the week.\n* *Relaxation techniques:* Relaxation techniques, such as yoga, meditation, and deep breathing, can help to calm the body and mind. Practice relaxation techniques for at least 10 minutes each day.\n* *Healthy diet:* Eating a healthy diet can help to improve overall health and well-being, which can in turn reduce anxiety levels. Aim for a diet that is rich in fruits, vegetables, and whole grains.\n* *Sleep:* Getting enough sleep is essential for both physical and mental health. Aim for 7-8 hours of sleep each night.\n* *Social support:* Talking to friends, family, or a therapist can help to provide support and reduce feelings of isolation. Spend time with people you care about, and talk to them about how you are feeling.\n* *Professional help:* If you are struggling to manage your anxiety on your own, it is important to seek professional help. A therapist can help you to identify the root of your anxiety and develop coping mechanisms.",
    
    "Is it normal to feel sad and unmotivated for no reason?": 
      "It is not uncommon to experience periods of sadness or lack of motivation without an apparent reason. However, if these feelings persist or significantly interfere with your daily life, it is important to consider the possibility of an underlying mental health condition, such as depression.",
    
    "How Can I control my Blood Pressure": 
      "Regular exercise can help control blood pressure. When you exercise, your heart rate and blood pressure increase. This helps to strengthen your heart and make it more efficient at pumping blood. As a result, your blood pressure will be lower when you are resting.\n\nIn addition to exercise, there are other things you can do to control your blood pressure, such as:\n\n* Eating a healthy diet\n* Maintaining a healthy weight\n* Quitting smoking\n* Reducing stress\n* Getting enough sleep",
  
    "How can I improve my overall physical well-being to ensure a healthy lifestyle?": 
      "To improve your overall physical well-being and ensure a healthy lifestyle, consider the following:\n\n*Physical Fitness:*\n\n Engage in regular physical activity, aiming for at least 150 minutes of moderate-intensity exercise or 75 minutes of vigorous-intensity exercise per week.\n* Choose activities you enjoy to make exercise sustainable.\n* Incorporate strength training exercises to build muscle mass and improve bone health.\n\n*Nutrition:*\n\n Consume a balanced diet rich in fruits, vegetables, whole grains, and lean protein.\n* Limit processed foods, sugary drinks, and unhealthy fats.\n* Stay hydrated by drinking plenty of water throughout the day.\n\n*Sleep:*\n\n Aim for 7-9 hours of quality sleep each night.\n* Establish a regular sleep-wake cycle, even on weekends.\n* Create a conducive sleep environment by making sure your bedroom is dark, quiet, and cool.\n\n*Stress Management:*\n\n Engage in stress-reducing activities such as exercise, yoga, meditation, or spending time in nature.\n* Practice deep breathing exercises to calm your mind and body.\n* Seek professional help if you experience chronic stress or anxiety.\n\n*Social Connections:*\n\n Build and maintain strong relationships with family, friends, and loved ones.\n* Participate in social activities and engage with your community.\n* Volunteer or engage in other activities that allow you to connect with others.\n\n*Regular Check-ups:*\n\n Visit your doctor for regular check-ups to monitor your overall health and identify any potential health issues early on.\n* Follow your doctor's recommendations for screenings, immunizations, and preventive care.\n\n*Mindset:*\n\n Cultivate a positive mindset and focus on your strengths.\n* Set realistic health goals and celebrate your progress.\n* Be patient and persistent, as improving your well-being is an ongoing journey.",
  
    "How Can I improve my mental health?": 
      "1. Identify the root of the issue:* Determine the underlying causes of your mental health concerns. This may involve seeking professional help from a therapist or counselor.\n\n*2. Practice mindfulness:* Engage in activities that promote present-moment awareness and reduce stress, such as yoga, meditation, or deep breathing exercises.\n\n*3. Engage in self-care activities:* Prioritize activities that nourish your mental and emotional well-being, such as spending time in nature, pursuing hobbies, or connecting with loved ones.\n\n*4. Build a support system:* Surround yourself with supportive individuals who understand and encourage your mental health journey.\n\n*5. Seek professional help when needed:* If your mental health concerns persist or worsen, don't hesitate to seek professional assistance from a therapist or counselor. They can provide personalized guidance and support tailored to your specific needs."
  };
  
  const submitHandler = async (e, prompt = null) => {
    // console.log(text);
    if (text != "") {
      e.preventDefault();

    }


    // setMessage(text);
    // setErrorText('');
    // return;
    if (!text) return;
    // let answer = answers[text.trim()];
    // answer = formatApiResponse(answer)
    // if (answer) {
    //   setMessage({
    //     role: 'Cure Me',
    //     content: answer,
    //   });
    //   setErrorText('');
    //   setTimeout(() => {
    //     scrollToLastItem.current?.scrollIntoView({ behavior: 'smooth' });
    //   }, 1);
    //   setTimeout(() => {
    //     setText('');
    //   }, 2);
    // } else {

      try {
       
        console.log("Text Is", text)
        const response = await axios.post('http://127.0.0.1:5000/qaretrival', { question:text }, {
          headers: {
            'Content-Type': 'application/json',
          },
        });
       
        if (response.status === 429) {
          return setErrorText('Too many requests, please try again later.');
        }

        const data = response.data;
        console.log(response)
        if (data.error) {
          setErrorText(data.error.message);
          setText('');
        } else {
          setErrorText('');
          setMessage({
            role: 'Cure Me',
            content: response.data.Message,
          });
          setTimeout(() => {
            scrollToLastItem.current?.scrollIntoView({ behavior: 'smooth' });
          }, 1);
          setTimeout(() => {
            setText('');
          }, 2);
        }
      } catch (e) {
        setErrorText(e.message);
        console.error(e);
      } finally {
        // setIsResponseLoading(false);
      }
    // setIsResponseLoading(true);

  };

  useLayoutEffect(() => {
    const handleResize = () => {
      setIsShowSidebar(window.innerWidth <= 640);
    };
    handleResize();

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    const storedChats = localStorage.getItem('previousChats');

    if (storedChats) {
      setLocalChats(JSON.parse(storedChats));
    }
  }, []);

  useEffect(() => {
    if (!currentTitle && text && message) {
      setCurrentTitle(text);
    }

    if (currentTitle && message) {
      const newChat = {
        title: currentTitle,
        role: 'user',
        content: text,
      };

      const responseMessage = {
        title: currentTitle,
        role: message.role,
        content: message.content,
      };

      setPreviousChats((prevChats) => [...prevChats, newChat, responseMessage]);
      setLocalChats((prevChats) => [...prevChats, newChat, responseMessage]);

      const updatedChats = [...localChats, newChat, responseMessage];
      localStorage.setItem('previousChats', JSON.stringify(updatedChats));

      setText(''); // Clear the text input
      setMessage(null); // Reset message to prevent re-triggering
    }
  }, [message, currentTitle, text]);

  const currentChat = (localChats || previousChats).filter(
    (prevChat) => prevChat.title === currentTitle
  );

  const uniqueTitles = Array.from(
    new Set(previousChats.map((prevChat) => prevChat.title).reverse())
  );

  const localUniqueTitles = Array.from(
    new Set(localChats.map((prevChat) => prevChat.title).reverse())
  ).filter((title) => !uniqueTitles.includes(title));

  const handlePromptClick = (prompt) => {
    console.log(prompt.description);
    // setMessage({
    //   role: 'Cure me',
    //   content: prompt.description,
    // });
    // setErrorText('');
  };

  return (
    <div className='container'>
      <section className={`sidebar ${isShowSidebar ? 'open' : ''}`}>
        <div className='sidebar-header' onClick={createNewChat} role='button'>
          <BiPlus size={20} />
          <button>New Chat</button>
        </div>
        <div className='sidebar-history'>
          {uniqueTitles.length > 0 && previousChats.length !== 0 && (
            <>
              <p>Ongoing</p>
              <ul>
                {uniqueTitles?.map((uniqueTitle, idx) => (
                  <li key={idx} onClick={() => backToHistoryPrompt(uniqueTitle)}>
                    {uniqueTitle}
                  </li>
                ))}
              </ul>
            </>
          )}
          {localUniqueTitles.length > 0 && localChats.length !== 0 && (
            <>
              <p>Previous</p>
              <ul>
                {localUniqueTitles?.map((uniqueTitle, idx) => (
                  <li key={idx} onClick={() => backToHistoryPrompt(uniqueTitle)}>
                    {uniqueTitle}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
        <div className='sidebar-info'>
          <div className='sidebar-info-upgrade'>
            <BiUser size={20} />
            <p>About</p>
          </div>
          <div className='sidebar-info-user'>
            <BiSolidUserCircle size={20} />
            <p>User</p>
          </div>
        </div>
      </section>

      <section className='main'>
        {!currentTitle && (
          <div className='empty-chat-container'>
            <img
              src='images/logo.png'
              width={45}
              height={45}
              alt='Cure Me'
            />
            <h1>Cure Me</h1>
            <h3>How can I help you today?</h3>
            <div className='prompt-cards-container'>
              {prompts.map((prompt, index) => (
                <PromptCard
                  key={index}
                  prompt={prompt}
                  onClick={() => handlePromptClick(prompt)}
                />
              ))}
            </div>
          </div>
        )}

        {isShowSidebar ? (
          <MdMenu
            className='hamburger'
            size={28.8}
            onClick={toggleSidebar}
          />
        ) : (
          <MdOutlineArrowLeft
            className='burger'
            size={28.8}
            onClick={toggleSidebar}
          />
        )}
        <div className='main-header'>
          <ul>
            {currentChat?.map((chatMsg, idx) => {
              const isUser = chatMsg.role === 'user';

              return (
                <li key={idx} ref={scrollToLastItem}>
                  {isUser ? (
                    <div>
                      <BiSolidUserCircle size={28.8} />
                    </div>
                  ) : (
                    <img src='images/logo.png' alt='Cure Me' />
                  )}
                  {isUser ? (
                    <div>
                      <p className='role-title'>You</p>
                      <p>{chatMsg.content}</p>
                    </div>
                  ) : (
                    <div>
                      <p className='role-title'>Cure Me</p>
                      <p>{chatMsg.content}</p>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
        <div className='main-bottom'>
          {errorText && <p className='errorText'>{errorText}</p>}
          {errorText && (
            <p id='errorTextHint'></p>
          )}
          <form className='form-container' onSubmit={submitHandler}>
            <input
              type='text'
              placeholder='Send a message.'
              spellCheck='false'
              value={isResponseLoading ? 'Processing...' : text}
              onChange={(e) => setText(e.target.value)}
              readOnly={isResponseLoading}
            />
            {!isResponseLoading && (
              <button type='submit'>
                <BiSend size={20} />
              </button>
            )}
          </form>
        </div>
      </section>
    </div>
  );
}

export default App;
