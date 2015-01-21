package utils;

import java.io.FileInputStream;
import java.util.MissingResourceException;
import java.util.PropertyResourceBundle;
import java.util.ResourceBundle;

public class SimpleConfiguration {
    protected ResourceBundle resources;

    protected String confFileName = "./config.properties";

    public SimpleConfiguration(){
        FileInputStream fileInputStream;
        try {
            fileInputStream = new FileInputStream(this.confFileName);
            this.resources = new PropertyResourceBundle(fileInputStream);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public String get(String key){
        try {
            String res = this.resources.getString(key);
            return res.trim();
        } catch (MissingResourceException e) {
            return null;
        } catch (Exception e){
            System.err.println("Can't get configuration value for the key " + key);
            System.exit(-1);
        }
        return key;
    }
    public static String valueOf(String key) {
        return new SimpleConfiguration().get(key);
    }
}
