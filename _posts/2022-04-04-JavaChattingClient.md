---
layout: single
title:  "자바 채팅프로그램 (클라이언트)"
categories: Java
tag: [java, chatting, threadPool, tcp/ip, thread]
toc: true
author_profile: true
toc_sticky: true
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


<p>threadPool을 활용한 자바 채팅 프로그램의 서버</p>


## 구성
- 클라이언트와 메세지를 주고 받는 Main.java
- 채팅 접속과 메세지를 입력, 송신하는 Start.java

## 소스코드


### Client.java


```
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;

import javax.swing.*;

public class Main extends JFrame{
	
	Socket socket;
	JTextArea textArea;
	
	// 클라이언트 프로그램의 작동을 시작하는 메소드
	public void startClient(String IP, int port) {
		Thread thread = new Thread() {
			public void run() {
				try {
					socket = new Socket(IP, port);
					receive();
				} catch(Exception e) {
					if(!socket.isClosed()) {
						stopClient();
						System.out.println("[서버 접속 실패]");
					}
				}
			}
		};
		thread.start();
	}
	
	// 클라이언트 프로그램의 작동을 종료하는 메소드
	public void stopClient() {
		try {
			if(socket != null && !socket.isClosed()) {
				socket.close();
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	// 서버로부터 메세지를 전달받는 메소드
	public void receive() {
		while(true) {
			try {
				InputStream in = socket.getInputStream();
				byte[] buffer = new byte[512];
				int length = in.read(buffer);
				if(length == -1) throw new IOException();
				String message = new String(buffer, 0, length, "UTF-8");
				textArea.append(message);
			} catch(Exception e) {
				stopClient();
				break;
			}
		}
	}
	
	// 서버로 메세지를 전송하는 메소드
	public void send(String message) {
		Thread thread = new Thread() {
			public void run() {
				try {
					OutputStream out = socket.getOutputStream();
					byte[] buffer = message.getBytes("UTF-8");
					out.write(buffer);
					out.flush();
				} catch(Exception e) {
					stopClient();
				}
			}
		};
		thread.start();
	}
	
	public Main() {
		setTitle("[채팅 클라이언트]");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		Container c = getContentPane();
		c.setLayout(new BorderLayout());
		
		JTextField userName = new JTextField("대화명을 입력하세요.");
		userName.setSize(150, 30);
		
		JTextField IPText = new JTextField("127.0.0.1");
		JTextField portText = new JTextField("9876");
		portText.setSize(80, 30);

		JPanel panel1 = new JPanel(new BorderLayout());
		c.add(panel1, BorderLayout.NORTH);
		panel1.add(userName, BorderLayout.WEST);
		panel1.add(IPText, BorderLayout.CENTER);
		panel1.add(portText, BorderLayout.EAST);
		
		JPanel panel3 = new JPanel(new BorderLayout());
		textArea = new JTextArea();
		textArea.setEditable(false);
		JScrollPane scrollPane = new JScrollPane(textArea);
		c.add(panel3, BorderLayout.CENTER);
		panel3.add(textArea, BorderLayout.CENTER);
		panel3.add(scrollPane, BorderLayout.EAST);
		
		JTextField input = new JTextField();
		input.setEnabled(false);
		input.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				JTextField t = (JTextField)e.getSource();
				send(userName.getText() + ": " + t.getText() + "\n");
				input.setText("");
				input.requestFocus();
			}
		});
		
		JButton sendButton = new JButton("보내기");
		sendButton.setEnabled(false);
		
		sendButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				send(userName.getText() + ": " + input.getText() + "\n");
				input.setText("");
				input.requestFocus();
			}
		});
		
		JButton connectionButton = new JButton("접속하기");
		connectionButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if(connectionButton.getText().equals("접속하기")) {
					int port = 9876;
					try {
						port = Integer.parseInt(portText.getText());
					} catch(Exception e2) {
						e2.printStackTrace();
					}
					startClient(IPText.getText(), port);
					textArea.append("[ 채팅방 접속 ]\n");
					connectionButton.setText("종료하기");
					input.setEnabled(true);
					sendButton.setEnabled(true);
					input.requestFocus();
				} else {
					stopClient();
					textArea.append("[ 채팅방 퇴장 ]\n");
					connectionButton.setText("접속하기");
					input.setEnabled(false);
					sendButton.setEnabled(false);
				}
			}
		});
		JPanel panel2 = new JPanel(new BorderLayout());
		c.add(panel2, BorderLayout.SOUTH);
		panel2.add(connectionButton, BorderLayout.WEST);
		panel2.add(input, BorderLayout.CENTER);
		panel2.add(sendButton, BorderLayout.EAST);
		
		setVisible(true);
		setSize(300, 300);
	}
	
	// 프로그램의 진입점
	public static void main(String[] args) {
		new Main();
	}
}
```


### Start.java
```
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class Start extends Main {
	public Start() {
		setTitle("[채팅 클라이언트]");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		Container c = getContentPane();
		c.setLayout(new BorderLayout());
		
		JTextField userName = new JTextField("대화명");
		userName.setSize(150, 30);
		
		JTextField IPText = new JTextField("127.0.0.1");
		JTextField portText = new JTextField("9876");
		portText.setSize(80, 30);

		JPanel panel1 = new JPanel(new BorderLayout());
		c.add(panel1, BorderLayout.NORTH);
		panel1.add(userName, BorderLayout.WEST);
		panel1.add(IPText, BorderLayout.CENTER);
		panel1.add(portText, BorderLayout.EAST);
		
//		JTextArea textArea = new JTextArea();
		textArea.setEditable(false);
		c.add(textArea, BorderLayout.CENTER);
		
		JTextField input = new JTextField();
		input.setEnabled(false);
		input.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				JTextField tmp = new JTextField();
				send(userName.getText() + ": " + tmp.getText() + "\n");
				input.setText("");
				input.requestFocus();
			}			
		});
		
		JButton sendButton = new JButton("보내기");
		sendButton.setEnabled(false);
		
		sendButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				send(userName.getText() + ": " + input.getText() + "\n");
				input.setText("");
				input.requestFocus();
			}
		});
		
		JButton connectionButton = new JButton("접속하기");
		connectionButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if(connectionButton.getText().equals("접속하기")) {
					int port = 9876;
					try {
						port = Integer.parseInt(portText.getText());
					} catch(Exception e2) {
						e2.printStackTrace();
					}
					startClient(IPText.getText(), port);
					textArea.append("[ 채팅방 접속 ]\n");
					connectionButton.setText("종료하기");
					input.setEnabled(true);
					sendButton.setEnabled(true);
					input.requestFocus();
				} else {
					stopClient();
					textArea.append("[ 채팅방 퇴장 ]\n");
					connectionButton.setText("접속하기");
					input.setEnabled(false);
					sendButton.setEnabled(false);
				}
			}
		});
		JPanel panel2 = new JPanel(new BorderLayout());
		c.add(panel2, BorderLayout.SOUTH);
		panel2.add(connectionButton, BorderLayout.WEST);
		panel2.add(input, BorderLayout.CENTER);
		panel2.add(sendButton, BorderLayout.EAST);
		
		setVisible(true);
		setSize(300, 300);
	}
}
```

