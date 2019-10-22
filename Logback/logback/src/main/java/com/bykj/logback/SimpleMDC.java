package com.bykj.logback;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;


public class SimpleMDC {
  static public void main(String[] args) throws Exception {

    // 你可以选择在任何时候将值放入 MDC 中    
    MDC.put("first", "Dorothy");

    Logger logger = LoggerFactory.getLogger(SimpleMDC.class);
    MDC.put("last", "Parker");

    //logger.info("Check enclosed.");
    //logger.debug("The most beautiful two words in English.");

    MDC.put("first", "Richard");
    MDC.put("last", "Nixon");
    //logger.info("I am not a crook.");
    logger.info("Attributed to the former US president. 17 Nov 1973.");
  }


}
