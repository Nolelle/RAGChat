import React, { useState } from "react";
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ChatWindow = ({ messages, isLoading, error, chatEndRef }) => {
    const [expandedSources, setExpandedSources] = useState({});
    
    const toggleSourceExpansion = (messageIndex) => {
        setExpandedSources(prev => ({
            ...prev,
            [messageIndex]: !prev[messageIndex]
        }));
    };
    
    // Function to extract filename without UUID prefix
    const formatFileName = (fileName) => {
        // Check if the filename has a UUID pattern (32 hex chars + underscore)
        const uuidPattern = /^[a-f0-9]{32}_(.+)$/i;
        const match = fileName.match(uuidPattern);
        return match ? match[1] : fileName;
    };
    
    return (
        <div className="flex-1 bg-gray-800 p-4 overflow-auto flex flex-col">
            {messages.length === 0 ? (
                <div className="text-gray-300 text-center mt-10 flex-1 flex items-center justify-center">
                    <div>
                        <h2 className="text-2xl font-bold mb-2 text-white">First Responders Chatbot</h2>
                        <p className="mb-4">Ask me anything about first aid or emergency response.</p>
                        <p className="text-sm text-gray-400">You can also upload documents for additional context.</p>
                    </div>
                </div>
            ) : (
                <div className="space-y-4">
                    {messages.map((msg, index) => (
                        <div 
                            key={index} 
                            className={`p-4 rounded-lg max-w-[80%] ${
                                msg.isUser 
                                    ? 'bg-blue-600 text-white ml-auto' 
                                    : 'bg-gray-700 text-gray-100 border border-gray-600'
                            }`}
                        >
                            <div className="text-current prose prose-invert prose-sm max-w-none">
                                {msg.isUser ? (
                                    msg.text
                                ) : (
                                    <ReactMarkdown
                                        components={{
                                            // Enhanced markdown rendering with code blocks
                                            code({node, inline, className, children, ...props}) {
                                                const match = /language-(\w+)/.exec(className || '');
                                                return !inline && match ? (
                                                    <SyntaxHighlighter
                                                        style={vscDarkPlus}
                                                        language={match[1]}
                                                        PreTag="div"
                                                        {...props}
                                                    >
                                                        {String(children).replace(/\n$/, '')}
                                                    </SyntaxHighlighter>
                                                ) : (
                                                    <code className={className} {...props}>
                                                        {children}
                                                    </code>
                                                );
                                            },
                                            // Enhance tables with better styling
                                            table({node, children, ...props}) {
                                                return (
                                                    <div className="overflow-x-auto my-4">
                                                        <table className="border-collapse border border-gray-600 w-full" {...props}>
                                                            {children}
                                                        </table>
                                                    </div>
                                                );
                                            },
                                            thead({node, children, ...props}) {
                                                return (
                                                    <thead className="bg-gray-800" {...props}>
                                                        {children}
                                                    </thead>
                                                );
                                            },
                                            th({node, children, ...props}) {
                                                return (
                                                    <th className="border border-gray-600 px-4 py-2 text-left font-bold" {...props}>
                                                        {children}
                                                    </th>
                                                );
                                            },
                                            td({node, children, ...props}) {
                                                return (
                                                    <td className="border border-gray-600 px-4 py-2" {...props}>
                                                        {children}
                                                    </td>
                                                );
                                            },
                                            // Better headings
                                            h1({node, children, ...props}) {
                                                return (
                                                    <h1 className="text-xl font-bold mt-6 mb-4 pb-2 border-b border-gray-600" {...props}>
                                                        {children}
                                                    </h1>
                                                );
                                            },
                                            h2({node, children, ...props}) {
                                                return (
                                                    <h2 className="text-lg font-bold mt-5 mb-3" {...props}>
                                                        {children}
                                                    </h2>
                                                );
                                            }
                                        }}
                                    >
                                        {msg.text}
                                    </ReactMarkdown>
                                )}
                            </div>
                            {msg.context && msg.context.length > 0 && (
                                <div className="mt-3 pt-2 border-t border-gray-600 text-xs text-gray-300">
                                    <div 
                                        className="font-semibold mb-2 flex items-center cursor-pointer hover:text-blue-300"
                                        onClick={() => toggleSourceExpansion(index)}
                                    >
                                        <svg 
                                            xmlns="http://www.w3.org/2000/svg" 
                                            className={`h-4 w-4 mr-1 transition-transform ${expandedSources[index] ? 'rotate-90' : ''}`} 
                                            fill="none" 
                                            viewBox="0 0 24 24" 
                                            stroke="currentColor"
                                        >
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                        </svg>
                                        Sources ({msg.context.length})
                                    </div>
                                    
                                    {expandedSources[index] && (
                                        <div className="mt-2 space-y-2 bg-gray-800 p-2 rounded">
                                            {msg.context.map((ctx, ctxIndex) => (
                                                <div 
                                                    key={ctxIndex} 
                                                    className="p-2 rounded bg-gray-750 border border-gray-600 hover:border-blue-500 transition-colors"
                                                >
                                                    <div className="font-medium text-blue-300 mb-1">
                                                        {formatFileName(ctx.file_name)}
                                                    </div>
                                                    <div className="text-gray-300 pl-2 border-l-2 border-blue-500">
                                                        {ctx.snippet}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
            
            {isLoading && (
                <div className="p-4 rounded-lg bg-gray-700 border border-gray-600 shadow-sm max-w-[80%] mt-4">
                    <div className="flex items-center">
                        <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce mr-2"></div>
                        <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce mr-2" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                        <span className="ml-2 text-gray-300">Thinking...</span>
                    </div>
                </div>
            )}
            
            {error && (
                <div className="p-3 bg-red-900 border border-red-700 text-red-100 rounded-lg mt-4 max-w-[80%]">
                    <p className="text-sm">{error}</p>
                </div>
            )}
            
            <div ref={chatEndRef} />
        </div>
    );
};

export default ChatWindow;