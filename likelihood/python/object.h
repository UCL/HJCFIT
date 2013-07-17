namespace {

    template<class T> class Object;
    // Helper function to create a python pointer that knows to decref itself on deconstruction.
    template<class T> Object<T> steal_ref(T * const _in); 
        
    // Takes ownership of python object and decrefs when deconstructed.
    // This object *steals* a reference to the PyObject. 
    template<class T = PyObject> class Object {
      friend Object<T> steal_ref<>(T * const);
      public:
        //! Constructs an object holding nothing.
        Object() : object_(NULL) {};
        //! \brief Destroys internal object
        //! \details Decrements refence count of object if it is non NULL.
        ~Object() {
          PyObject * const dummy = (PyObject*)object_;
          object_ = NULL;
          Py_XDECREF(((PyObject*)dummy));
        }
        //! Acquires reference from another object.
        Object(Object const &_c) : object_(_c.object_) { Py_XINCREF(object_); }
        //! Steals reference from another object.
        Object(Object && _c) : object_(_c.object_) { _c.object_ = NULL; }
        //! Assignement operation.
        bool operator=(Object const &_c) { return acquire(_c.object_); }
        //! Move assignement operation.
        void operator=(Object &&_c) {
          if(&_c != this) return steal(_c.release());
          return is_valid();
        }
        //! Returns a borrowed reference to the internal object.
        T* operator ~() const { return object_; }
        //! True if internal object is not NULL.
        bool operator!() const { return object_ == NULL; }
        //! True if internal object is not NULL.
        bool is_valid() const { return object_ != NULL; }
        //! \brief Returns object pointer without decreasing reference count. 
        //! \details The caller will own the return object. It is the caller's responsibility to
        //! decrease the reference count of the object if it non null.
        T* new_ref() const { Py_INCREF( ((PyObject*)object_) ); return object_; }
        //! \brief Returns object pointer and sets internal object to NULL.
        //! \details The caller will own the return object. It is the caller's responsibility to
        //! decrease the reference count of the object if it non null.
        T* release() { T* const dummy(object_); object_ = NULL; return dummy; }
        //! \brief Steals a reference to a new object pointer.
        //! \details Decreases reference count of *current* object (if not NULL) by one. The new
        //!          object's reference count is *not* increased.
        //! \returns True if _in is not NULL.
        bool steal(T* _in) {
          std::swap(object_, _in);
          Py_XDECREF(((PyObject*)_in));
          return object_ != NULL;
        }
        //! \brief Acrquores a reference to a new object pointer.
        //! \details Decreases reference count of *current* object (if not NULL) by one. The new
        //!          object's reference count is increased by one (if not NULL).
        //! \returns True if _in is not NULL.
        bool acquire(T* _in) {
          Py_XINCREF(_in);
          return steal(_in); 
        }
        operator bool() const { return object_ != NULL; }

      private:
        explicit Object(T * const _in) : object_(_in) {}
        T * object_; 
    };

    template<class T> Object<T> steal_ref(T * const _in) { return Object<T>(_in); }
}