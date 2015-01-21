package utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

public class OwnConfiguration {

    public static String configurationDir = ".";

    protected String confFileName = "./config.properties";
    protected Properties properties = new Properties();

    public OwnConfiguration(String fileName) {
        try {
            if (fileName == null || fileName.equals("")) {
                fileName = this.confFileName;
            } else {
                this.confFileName = fileName;
            }
            File f = new File(OwnConfiguration.configurationDir + fileName);
            if (!f.createNewFile()) {
                this.properties.load(new FileInputStream(f));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void load() throws IOException{
        File f = new File(OwnConfiguration.configurationDir + this.confFileName);
        if (!f.createNewFile()) {
            this.properties.load(new FileInputStream(f));
        }
    }

    public String get(String key) {
        try {
            this.load();
            return this.properties.getProperty(key);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return key;
    }

    public String valueOf(String key){
        try {
            this.load();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return this.get(key);
    }

    public void store() throws IOException {
        this.properties.store(new FileOutputStream(new File(OwnConfiguration.configurationDir +
                this.confFileName)), "");
    }

    public void setValue(String key, String value) throws IOException {
        this.properties.setProperty(key, value);
        this.properties.store(new FileOutputStream(new File(OwnConfiguration.configurationDir +
                this.confFileName)), "");
    }

    public void setValue(String key, String[] value) throws IOException {
        String values = "";
        // Make splitted values
        for (String currentValue: value) {
            values += currentValue + ",";
        }
        // Finally, store the value
        properties.setProperty(key, values);
        properties.store(new FileOutputStream(new File(OwnConfiguration.configurationDir +
                this.confFileName)), "");
    }

    public String[] getArray(String key) {
        String value = get(key);
        return value.split(",");
    }

    public int getInt(String key) {
        return (Integer.parseInt(get(key)));
    }

    public double getDouble(String key) {
        return (Double.parseDouble(get(key)));
    }
}