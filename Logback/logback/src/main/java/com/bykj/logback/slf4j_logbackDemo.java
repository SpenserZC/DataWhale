package com.bykj.logback;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class slf4j_logbackDemo {
    Logger logger=  LoggerFactory.getLogger(slf4j_logbackDemo.class);

    @Test
    public void test() {
        logger.debug("debug message");
        logger.info("info message");
        logger.warn("warning message");
        logger.error("error message");
        logger.warn("login message");
    }
}
